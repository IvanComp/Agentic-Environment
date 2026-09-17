from __future__ import annotations

import unittest

from agentic_quant.install_cuda import (
    cuda_runtime_packages,
    parse_driver_cuda_version,
    select_cuda_wheel,
)


class CudaInstallerTest(unittest.TestCase):
    def test_parses_nvidia_smi_cuda_version(self) -> None:
        output = "NVIDIA-SMI 570.00  Driver Version: 570.00  CUDA Version: 12.8"
        self.assertEqual(parse_driver_cuda_version(output), (12, 8))

    def test_selects_latest_compatible_cuda_wheel(self) -> None:
        self.assertEqual(select_cuda_wheel((11, 8)), "cu118")
        self.assertEqual(select_cuda_wheel((12, 8)), "cu125")
        self.assertEqual(select_cuda_wheel((13, 1)), "cu130")
        self.assertEqual(select_cuda_wheel((13, 2)), "cu132")

    def test_rejects_driver_older_than_available_wheels(self) -> None:
        with self.assertRaisesRegex(SystemExit, "oldest available wheel"):
            select_cuda_wheel((11, 7))

    def test_selects_runtime_packages_for_cuda_major_version(self) -> None:
        self.assertEqual(
            cuda_runtime_packages("cu125"),
            [
                "nvidia-cuda-runtime-cu12",
                "nvidia-cublas-cu12",
                "nvidia-nvjitlink-cu12",
            ],
        )
        self.assertEqual(
            cuda_runtime_packages("cu118"),
            ["nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11"],
        )


if __name__ == "__main__":
    unittest.main()
