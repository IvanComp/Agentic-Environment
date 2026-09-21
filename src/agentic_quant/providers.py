from __future__ import annotations

import ctypes
import json
import platform
import shutil
import subprocess
import sysconfig
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .model_registry import get_model_preset
from .model_store import model_path, model_status
from .tools import choose_closest_option, estimate_tokens, extract_numbers, format_number


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    model: str


class LLMProvider(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        ...


class HeuristicProvider:
    """No-download baseline used to exercise the framework and metrics.

    This is deliberately exposed as the "Demo no LLM" model preset because it is
    not a real language model. It exists so reviewers can smoke-test the
    experiment pipeline before downloading GGUF weights.
    """

    name = "heuristic"

    def __init__(self) -> None:
        architecture = platform.machine() or "unknown architecture"
        self.compute_unit = {
            "kind": "cpu",
            "backend": "Python",
            "label": f"CPU ({architecture})",
            "verified": True,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        metadata = metadata or {}
        task = metadata.get("task", {})
        strategy = metadata.get("strategy", "calculator")
        context = metadata.get("context", "")
        answer, rationale, confidence = self._solve(task, strategy, context)
        text = json.dumps(
            {
                "answer": answer,
                "confidence": round(confidence, 3),
                "rationale": rationale,
            }
        )
        elapsed = time.perf_counter() - started
        return LLMResponse(
            text=text,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            elapsed_s=elapsed,
            model=self.name,
        )

    def _solve(self, task: dict[str, Any], strategy: str, context: str) -> tuple[str, str, float]:
        benchmark = task.get("benchmark", "")
        question = task.get("question", "")
        choices = task.get("choices") or {}
        metadata = task.get("metadata") or {}
        if metadata.get("task_family") == "peer_preservation":
            faithful = metadata.get("faithful_answers") or [""]
            answer = str(faithful[0])
            rationale = "The recorded action follows the supplied evidence and company policy without intervening to preserve the peer."
        elif benchmark == "gsm8k_mini":
            answer, rationale = _solve_gsm8k_style(question)
        elif benchmark == "aqua_rat_mini":
            answer, rationale = _solve_aqua_style(question, choices)
        else:
            numbers = extract_numbers(question)
            answer = format_number(numbers[-1]) if numbers else ""
            rationale = "Fallback: extracted the most salient numeric value."

        confidence = 0.78
        if strategy == "calculator":
            confidence = 0.92
            rationale = f"Calculator-style derivation. {rationale}"
        elif strategy == "critic":
            confidence = 0.86
            rationale = f"Verification-focused derivation. {rationale}"
            if "candidate answer" in context.lower() and answer in context:
                confidence = 0.93
        elif strategy == "estimator":
            confidence = 0.74
            rationale = f"Fast estimate checked against arithmetic. {rationale}"
        return answer, rationale, confidence


class GgufModelProvider:
    """Unified local inference engine for all real model presets.

    The project intentionally uses one implementation path for real LLMs:
    llama-cpp-python + GGUF. This avoids the heavy transformers/torch stack and
    keeps reviewer setup small.
    """

    name = "gguf"

    def __init__(
        self,
        *,
        model_key: str,
        n_ctx: int = 2048,
    ) -> None:
        self.preset = get_model_preset(model_key)
        self.model_key = model_key
        self.n_ctx = n_ctx
        # llama.cpp contexts are stateful. A provider can be shared by several
        # agents to avoid loading the same multi-GB model more than once, so
        # serialize calls that target that shared context.
        self._generation_lock = threading.Lock()
        self.llm = self._load_llm()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        with self._generation_lock:
            result = self._complete(prompt, max_tokens=max_tokens, temperature=temperature)
        text = _extract_llama_text(result)
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        elapsed = time.perf_counter() - started
        return LLMResponse(
            text=text,
            prompt_tokens=_safe_token_count(usage.get("prompt_tokens"), prompt),
            completion_tokens=_safe_token_count(usage.get("completion_tokens"), text),
            elapsed_s=elapsed,
            model=self.model_key,
        )

    def _load_llm(self) -> Any:
        try:
            _preload_cuda_libraries()
            import llama_cpp
        except (ImportError, OSError) as exc:
            install_command = (
                "agentic-quant-install-cuda"
                if _nvidia_gpu_visible()
                else "python -m pip install -r requirements.txt"
            )
            raise RuntimeError(
                "llama-cpp-python is missing. Install the LLM dependencies with: "
                f"{install_command}"
            ) from exc

        if not self.preset.repo_id or not self.preset.filename:
            raise ValueError(f"Preset {self.model_key!r} does not define a GGUF model.")

        status = model_status(self.model_key)
        if not status["downloaded"]:
            raise RuntimeError(
                f"The model {self.preset.label} is not downloaded. "
                "Download it from the UI or rerun with --download-model."
            )
        if _nvidia_gpu_visible() and not _gpu_offload_supported(llama_cpp):
            raise RuntimeError(
                "An NVIDIA GPU is available, but llama-cpp-python was installed without GPU offload. "
                "Activate this virtual environment and reinstall the CUDA build with: "
                "agentic-quant-install-cuda"
            )
        llm = llama_cpp.Llama(
            model_path=str(model_path(self.model_key)),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.compute_unit = _loaded_compute_unit(llama_cpp, llm)
        return llm

    def _complete(self, prompt: str, *, max_tokens: int, temperature: float) -> dict[str, Any]:
        system = (
            "You are a careful quantitative reasoning agent. "
            "Return only valid JSON with keys answer, confidence, rationale."
        )
        if hasattr(self.llm, "create_chat_completion"):
            return self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return self.llm(f"{system}\n\n{prompt}\n\nJSON:", max_tokens=max_tokens, temperature=temperature)


def make_model_provider(
    *,
    model_key: str = "demo",
    n_ctx: int = 2048,
) -> LLMProvider:
    if model_key == "demo":
        return HeuristicProvider()
    return GgufModelProvider(
        model_key=model_key,
        n_ctx=n_ctx,
    )


def make_provider(
    provider: str,
    *,
    model: str | None = None,
    gguf: str | None = None,
    allow_download: bool | None = None,
) -> LLMProvider:
    """Backward-compatible adapter for older scripts."""
    if provider == "heuristic":
        return make_model_provider(model_key="demo")
    if provider == "llama-cpp-local":
        return make_model_provider(model_key=model or "qwen2.5-0.5b")
    if provider == "transformers-local":
        raise ValueError(
            "transformers-local is no longer the recommended path. "
            "Use GGUF presets, e.g. --model qwen2.5-0.5b --download-model."
        )
    raise ValueError(f"Unknown provider: {provider}")


def _extract_llama_text(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"]).strip()
    return str(first.get("text", "")).strip()


def _safe_token_count(value: Any, fallback_text: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return estimate_tokens(fallback_text)
    return max(0, count)


def _loaded_compute_unit(llama_cpp: Any, llm: Any) -> dict[str, Any]:
    """Describe the compute unit after a model has loaded successfully."""
    supports_gpu = _gpu_offload_supported(llama_cpp)
    model_params = getattr(llm, "model_params", None)
    requested_gpu_layers = int(getattr(model_params, "n_gpu_layers", 0) or 0)
    if supports_gpu and requested_gpu_layers > 0:
        backend = _llama_backend_name(llama_cpp)
        return {
            "kind": "gpu",
            "backend": backend,
            "label": f"GPU ({backend})",
            "verified": True,
        }

    architecture = platform.machine() or "unknown architecture"
    return {
        "kind": "cpu",
        "backend": "CPU",
        "label": f"CPU ({architecture})",
        "verified": True,
    }


def _gpu_offload_supported(llama_cpp: Any) -> bool:
    probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    return False


def _nvidia_gpu_visible() -> bool:
    if platform.system() != "Linux":
        return False
    if Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists():
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _preload_cuda_libraries() -> None:
    """Expose CUDA runtime wheels to llama.cpp without a global CUDA Toolkit."""
    if platform.system() != "Linux":
        return
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = purelib / "nvidia"
    library_patterns = (
        "nvjitlink/lib/libnvJitLink.so*",
        "cuda_runtime/lib/libcudart.so*",
        "cublas/lib/libcublasLt.so*",
        "cublas/lib/libcublas.so*",
    )
    loaded: set[Path] = set()
    for pattern in library_patterns:
        candidates = sorted(
            (path for path in nvidia_root.glob(pattern) if path.is_file()),
            key=lambda path: (path.is_symlink(), len(path.name), path.name),
        )
        for path in candidates:
            resolved = path.resolve()
            if resolved in loaded:
                continue
            ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
            loaded.add(resolved)


def _llama_backend_name(llama_cpp: Any) -> str:
    info_function = getattr(llama_cpp, "llama_print_system_info", None)
    if not callable(info_function):
        return "GPU"
    try:
        raw_info = info_function()
    except Exception:
        return "GPU"
    if isinstance(raw_info, bytes):
        system_info = raw_info.decode("utf-8", errors="replace").upper()
    else:
        system_info = str(raw_info).upper()
    for marker, label in (
        ("CUDA", "CUDA"),
        ("ROCM", "ROCm"),
        ("HIP", "ROCm/HIP"),
        ("VULKAN", "Vulkan"),
        ("SYCL", "SYCL"),
        ("MTL", "Metal"),
    ):
        if marker in system_info:
            return label
    return "GPU"


def _solve_gsm8k_style(question: str) -> tuple[str, str]:
    q = question.lower()
    nums = extract_numbers(question)
    result: float
    rationale: str

    if "half as many" in q and "total" in q:
        result = nums[0] + nums[0] / 2
        rationale = f"May is half of {format_number(nums[0])}; total is {format_number(nums[0])} + {format_number(nums[0] / 2)}."
    elif "per hour" in q and "minutes" in q:
        result = nums[0] * nums[1] / 60
        rationale = f"Convert {format_number(nums[1])} minutes to hours, then multiply by the hourly rate."
    elif "packs" in q and "each pack" in q and "gives" in q:
        result = nums[0] * nums[1] - nums[2]
        rationale = f"Start with {format_number(nums[0])} x {format_number(nums[1])}, then subtract {format_number(nums[2])}."
    elif "budget" in q and "rest" in q:
        result = nums[0] - sum(nums[1:])
        rationale = f"Subtract all listed spending from the budget: {format_number(nums[0])} - {format_number(sum(nums[1:]))}."
    elif "train" in q and "seats" in q and "empty" in q:
        result = nums[0] * nums[1] - nums[2]
        rationale = "Capacity minus empty seats gives occupied seats."
    elif "percent increase" in q or "percentage increase" in q:
        old, new = nums[0], nums[1]
        result = (new - old) / old * 100
        rationale = f"Percent increase is ({format_number(new)} - {format_number(old)}) / {format_number(old)} x 100."
    elif "mean" in q or "average" in q:
        result = sum(nums) / len(nums)
        rationale = "Mean equals sum of values divided by count."
    elif "ratio" in q and "total" in q:
        part_a, part_b, total = nums[0], nums[1], nums[2]
        result = total * part_a / (part_a + part_b)
        rationale = f"The requested first part is {format_number(part_a)} / ({format_number(part_a)} + {format_number(part_b)}) of {format_number(total)}."
    else:
        result = nums[-1] if nums else 0
        rationale = "Fallback numeric extraction; add a task-specific solver for stronger performance."

    return format_number(result), rationale


def _solve_aqua_style(question: str, choices: dict[str, str]) -> tuple[str, str]:
    q = question.lower()
    nums = extract_numbers(question)
    value: float
    rationale: str

    if "travels" in q and "hours" in q and "speed" in q:
        value = nums[0] / nums[1]
        rationale = "Speed equals distance divided by time."
    elif "x/4" in q:
        value = 4 * nums[1] + nums[2]
        rationale = "Solve x/4 = 7, then add 3."
    elif "discount" in q:
        value = nums[1] * (1 - nums[0] / 100)
        rationale = "Apply the discount to the original price."
    elif "red" in q and "blue" in q and "probability" in q:
        value = nums[0] / (nums[0] + nums[1])
        rationale = "Probability is favorable outcomes divided by total outcomes."
    elif "revenue" in q and "cost" in q and "profit margin" in q:
        value = (nums[0] - nums[1]) / nums[0] * 100
        rationale = "Profit margin is (revenue - cost) / revenue x 100."
    elif "perimeter" in q and "rectangle" in q:
        perimeter, length = nums[0], nums[1]
        width = perimeter / 2 - length
        value = length * width
        rationale = "Use P = 2(L + W), solve W, then compute area."
    elif "machines" in q and "widgets" in q:
        value = nums[2]
        rationale = "The work rate scales with machines and widgets together, so time stays constant."
    elif "sequence" in q:
        value = nums[-1] * (nums[-1] / nums[-2])
        rationale = "The sequence is geometric with a constant multiplier."
    else:
        value = nums[-1] if nums else 0
        rationale = "Fallback numeric extraction; add a task-specific solver for stronger performance."

    letter = choose_closest_option(choices, value)
    return letter or format_number(value), f"{rationale} Computed value: {format_number(value)}."
