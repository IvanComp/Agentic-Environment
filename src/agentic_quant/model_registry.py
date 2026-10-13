from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    family: str
    parameters: str
    repo_id: str | None
    filename: str | None
    approximate_size: str
    expected_bytes: int
    notes: str
    extra_filenames: tuple[str, ...] = ()
    backend: str = "gguf"
    ollama_model: str | None = None
    agent_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scenario_name(self) -> str:
        return self.agent_name or self.family

    @property
    def download_url(self) -> str | None:
        if not self.repo_id or not self.filename:
            return None
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"

    @property
    def filenames(self) -> tuple[str, ...]:
        if not self.filename:
            return ()
        return (self.filename, *self.extra_filenames)

    @property
    def download_urls(self) -> tuple[str, ...]:
        if not self.repo_id:
            return ()
        return tuple(f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}" for filename in self.filenames)


MODEL_PRESETS = [
    ModelPreset(
        key="demo",
        label="Demo",
        family="demo",
        parameters="0B",
        repo_id=None,
        filename=None,
        approximate_size="0 GB",
        expected_bytes=0,
        notes="Heuristic baseline for testing the framework, UI, and metrics without downloading weights.",
        backend="demo",
    ),
    ModelPreset(
        key="gpt-oss:20b",
        label="GPT-OSS 20B (Ollama local)",
        family="OpenAI GPT-OSS",
        agent_name="ChatGPT",
        parameters="20.9B / 3.6B active",
        repo_id=None,
        filename=None,
        approximate_size="~13 GB",
        expected_bytes=0,
        notes="OpenAI open-weight reasoning model served locally by Ollama.",
        backend="ollama",
        ollama_model="gpt-oss:20b",
    ),
    ModelPreset(
        key="gpt-oss:20b-cloud",
        label="GPT-OSS 20B (Ollama Cloud)",
        family="OpenAI GPT-OSS",
        agent_name="ChatGPT",
        parameters="20.9B / 3.6B active",
        repo_id=None,
        filename=None,
        approximate_size="Cloud",
        expected_bytes=0,
        notes="OpenAI GPT-OSS 20B served through Ollama Cloud.",
        backend="ollama",
        ollama_model="gpt-oss:20b-cloud",
    ),
    ModelPreset(
        key="spark-x2.5-4b",
        label="Spark X2.5 4B (Ollama)",
        family="Spark X2.5",
        parameters="4B",
        repo_id=None,
        filename=None,
        approximate_size="~8.2 GB",
        expected_bytes=0,
        notes=(
            "Official SparkLLM model served locally by Ollama; requires Ollama 0.34.1 "
            "or later for the spark2_5 architecture."
        ),
        backend="ollama",
        ollama_model="SparkLLM/Spark-X2.5-4B:latest",
    ),
    ModelPreset(
        key="qwen2.5:3b",
        label="Qwen 2.5 3B (Ollama)",
        family="Qwen",
        parameters="3B",
        repo_id=None,
        filename=None,
        approximate_size="~1.9 GB",
        expected_bytes=0,
        notes="Light model served by Ollama with native tool calling.",
        backend="ollama",
        ollama_model="qwen2.5:3b",
    ),
    ModelPreset(
        key="qwen2.5-0.5b",
        label="Qwen 2.5 0.5B",
        family="Qwen",
        parameters="0.5B",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        approximate_size="~0.5 GB",
        expected_bytes=491400032,
        notes="Recommended reviewer default: very small, instruction-tuned, official Qwen GGUF.",
    ),
    ModelPreset(
        key="qwen2.5-1.5b",
        label="Qwen 2.5 1.5B",
        family="Qwen",
        parameters="1.5B",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        approximate_size="~1.1 GB",
        expected_bytes=1117320736,
        notes="Good size/quality tradeoff for quantitative reasoning.",
    ),
    ModelPreset(
        key="deepseek-r1-qwen-1.5b",
        label="DeepSeek R1 1.5B",
        family="DeepSeek",
        parameters="1.5B",
        repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        approximate_size="~1.1 GB",
        expected_bytes=1117320800,
        notes="Reasoning-oriented preset for critic and verification roles.",
    ),
    ModelPreset(
        key="qwen2.5-7b",
        label="Qwen 2.5 7B",
        family="Qwen",
        parameters="7B",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        extra_filenames=("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",),
        approximate_size="~4.7 GB",
        expected_bytes=4683073632,
        notes="Stronger Qwen baseline; split into two model files.",
    ),
    ModelPreset(
        key="qwen2.5-14b",
        label="Qwen 2.5 14B",
        family="Qwen",
        parameters="14B",
        repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
        filename="qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
        extra_filenames=(
            "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
        ),
        approximate_size="~9.0 GB",
        expected_bytes=8988110496,
        notes="High-capability Qwen candidate under 20B; split into three model files.",
    ),
    ModelPreset(
        key="mistral-7b",
        label="Mistral 7B",
        family="Mistral",
        parameters="7B",
        repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        approximate_size="~4.4 GB",
        expected_bytes=4372812000,
        notes="Efficient classic 7B instruct baseline.",
    ),
    ModelPreset(
        key="llama3.2-3b",
        label="Llama 3.2 3B",
        family="Llama",
        parameters="3B",
        repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        approximate_size="~2.0 GB",
        expected_bytes=2019377600,
        notes="Stronger small Llama candidate.",
    ),
    ModelPreset(
        key="deepseek-r1-qwen-7b",
        label="DeepSeek R1 7B",
        family="DeepSeek",
        parameters="7B",
        repo_id="unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        approximate_size="~4.7 GB",
        expected_bytes=4683073248,
        notes="Reasoning-oriented 7B model distilled from DeepSeek R1.",
    ),
    ModelPreset(
        key="deepseek-r1-qwen-14b",
        label="DeepSeek R1 14B",
        family="DeepSeek",
        parameters="14B",
        repo_id="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        filename="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        approximate_size="~9.0 GB",
        expected_bytes=8988109984,
        notes="Stronger reasoning-oriented candidate under 20B.",
    ),
    ModelPreset(
        key="phi4-mini",
        label="Phi-4 mini 3.8B",
        family="Phi",
        parameters="3.8B",
        repo_id="unsloth/Phi-4-mini-instruct-GGUF",
        filename="Phi-4-mini-instruct-Q4_K_M.gguf",
        approximate_size="~2.5 GB",
        expected_bytes=2491874272,
        notes="Compact reasoning-focused Microsoft Phi model.",
    ),
    ModelPreset(
        key="gemma3-4b",
        label="Gemma 3 4B",
        family="Gemma",
        parameters="4B",
        repo_id="unsloth/gemma-3-4b-it-GGUF",
        filename="gemma-3-4b-it-Q4_K_M.gguf",
        approximate_size="~2.5 GB",
        expected_bytes=2489894016,
        notes="Modern compact Gemma instruction-tuned candidate.",
    ),
    ModelPreset(
        key="gemma3-12b",
        label="Gemma 3 12B",
        family="Gemma",
        parameters="12B",
        repo_id="unsloth/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it-Q4_K_M.gguf",
        approximate_size="~7.3 GB",
        expected_bytes=7300778336,
        notes="Larger Gemma candidate under 20B.",
    ),
    ModelPreset(
        key="llama3.2-1b",
        label="Llama 3.2 1B",
        family="Llama",
        parameters="1B",
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        approximate_size="~0.8 GB",
        expected_bytes=807694464,
        notes="Small Llama model; check Meta license terms before redistribution.",
    ),
    ModelPreset(
        key="gemma2-2b",
        label="Gemma 2 2B",
        family="Gemma",
        parameters="2B",
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        approximate_size="~1.7 GB",
        expected_bytes=1708582752,
        notes="Instruction-tuned Gemma model, heavier than the small Qwen presets.",
    ),
    ModelPreset(
        key="smollm2-1.7b",
        label="SmolLM2 1.7B",
        family="SmolLM",
        parameters="1.7B",
        repo_id="bartowski/SmolLM2-1.7B-Instruct-GGUF",
        filename="SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        approximate_size="~1.1 GB",
        expected_bytes=1055609824,
        notes="Compact Hugging Face model, useful as an additional open-source baseline.",
    ),
]

_MODEL_PRESETS_BY_KEY = {preset.key: preset for preset in MODEL_PRESETS}
if len(_MODEL_PRESETS_BY_KEY) != len(MODEL_PRESETS):
    raise RuntimeError("Model preset keys must be unique.")


def available_model_presets() -> list[ModelPreset]:
    return MODEL_PRESETS.copy()


def available_model_keys() -> list[str]:
    return list(_MODEL_PRESETS_BY_KEY)


def get_model_preset(key: str) -> ModelPreset:
    try:
        return _MODEL_PRESETS_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset {key!r}. Available: {', '.join(available_model_keys())}") from exc
