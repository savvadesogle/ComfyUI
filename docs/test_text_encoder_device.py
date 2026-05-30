"""Unit tests for text_encoder_initial_device().

Tests the decision logic independently of actual hardware by mocking
get_free_memory() and configuration flags.
"""
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, ".")

from comfy.model_management import text_encoder_initial_device, text_encoder_device
import torch


# ── helpers ───────────────────────────────────────────────────────────

def patch_mem(**kwargs):
    """Apply all required patches for a single test call."""
    # Default values mirror real hardware
    defaults = {
        "comfy.model_management.get_free_memory": lambda dev: 12.0 * 1024**3,
        "comfy.memory_management.aimdo_enabled": False,
        "comfy.model_management.is_device_mps": lambda dev: False,
        "comfy.model_management.args.gpu_only": False,
        "comfy.model_management.vram_state": type("VS", (), {"name": "NORMAL_VRAM"})(),
    }
    defaults.update(kwargs)

    mgr = patch.multiple("comfy.model_management", **{
        k: v if callable(v) else MagicMock(return_value=v)
        for k, v in defaults.items()
    })
    return mgr


def call_initial(load_dev="xpu:0", offload_dev="cpu", model_size_gb=7.6):
    """Call text_encoder_initial_device with convenience GB → bytes conversion."""
    return text_encoder_initial_device(
        torch.device(load_dev),
        torch.device(offload_dev),
        int(model_size_gb * 1024**3),
    )


# ── tests ─────────────────────────────────────────────────────────────

class TestTextEncoderInitialDevice:
    """Direct unit tests — mocks hardware, tests decision logic."""

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    @patch("comfy.memory_management.aimdo_enabled", True)
    def test_aimdo_returns_offload(self, *_):
        """When aimdo_enabled is True, always return offload_device."""
        dev = text_encoder_initial_device(
            torch.device("xpu:0"), torch.device("cpu"), 7_600_000_000
        )
        assert dev.type == "cpu"

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    def test_load_equals_offload(self, *_):
        """When load == offload, return offload immediately."""
        dev = text_encoder_initial_device(
            torch.device("xpu:0"), torch.device("xpu:0"), 7_600_000_000
        )
        assert dev.type == "xpu"

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    def test_small_model_returns_offload(self, *_):
        """Models ≤ 1 GB always go to CPU (offload_device)."""
        # 800 MB — well under 1 GB
        dev = text_encoder_initial_device(
            torch.device("xpu:0"), torch.device("cpu"), 800_000_000
        )
        assert dev.type == "cpu"

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    def test_1gb_model_returns_offload(self, *_):
        """Exactly 1 GB → still offload (≤ not <)."""
        dev = text_encoder_initial_device(
            torch.device("xpu:0"), torch.device("cpu"), 1024**3
        )
        assert dev.type == "cpu"

    def test_mps_returns_load(self):
        """MPS devices return load_device early via is_device_mps()."""
        with patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3):
            with patch("comfy.model_management.is_device_mps", return_value=True):
                dev = text_encoder_initial_device(
                    torch.device("mps:0"), torch.device("cpu"), 7_600_000_000
                )
                assert dev.type == "mps"

    # ── VRAM-sufficient cases ───────────────────────────────────────

    @pytest.mark.parametrize("free_gb,model_gb,expected", [
        (20,  7.6,  "xpu"),   # fits easily
        (15,  7.6,  "xpu"),   # 15 > 9.12 → GPU
        (32,  8.5,  "xpu"),   # fits
        (24,  15.0, "xpu"),   # 24 > 18 → GPU
        (24,  15.2, "xpu"),   # 24 > 18.24 → GPU
        (48,  15.2, "xpu"),   # fits
        (9.13, 7.6, "xpu"),   # boundary: 9.13 > 9.12 → GPU
    ])
    def test_fits_on_gpu(self, free_gb, model_gb, expected):
        """When model * 1.2 < free_vram, the fixed code returns GPU."""
        with patch("comfy.model_management.get_free_memory",
                   return_value=int(free_gb * 1024**3)):
            dev = text_encoder_initial_device(
                torch.device("xpu:0"), torch.device("cpu"),
                int(model_gb * 1024**3),
            )
            assert dev.type == expected

    # ── VRAM-insufficient cases ─────────────────────────────────────

    @pytest.mark.parametrize("free_gb,model_gb,expected", [
        (8,   7.6,  "cpu"),   # 9.12 > 8 → CPU
        (12,  15.0, "cpu"),   # 18 > 12 → CPU
        (12,  15.2, "cpu"),   # 18.24 > 12 → CPU
        (8,   8.5,  "cpu"),   # 10.2 > 8 → CPU
        (9.11, 7.6, "cpu"),   # boundary: 9.12 > 9.11 → CPU
        (0,   7.6,  "cpu"),   # no VRAM → CPU
    ])
    def test_does_not_fit(self, free_gb, model_gb, expected):
        """When model * 1.2 ≥ free_vram, return CPU."""
        with patch("comfy.model_management.get_free_memory",
                   return_value=int(free_gb * 1024**3)):
            dev = text_encoder_initial_device(
                torch.device("xpu:0"), torch.device("cpu"),
                int(model_gb * 1024**3),
            )
            assert dev.type == expected

    # ── regression guard: original vs fixed ─────────────────────────

    @pytest.mark.parametrize("free_vram_gb,free_ram_gb,model_gb", [
        # Cases where original said GPU (free_vram > free_ram/2 AND model fits)
        (20,  12,  7.6),    # 20 > 6  AND 9.12 < 20 → GPU → GPU
        (15,  20,  7.6),    # 15 > 10 AND 9.12 < 15 → GPU → GPU
        (32,  40,  15.0),   # 32 > 20 AND 18 < 32  → GPU → GPU
        # Cases where original said CPU but model fits (the bug)
        (15,  64,  7.6),    # 15 > 32? NO but 9.12 < 15 → was CPU, now GPU
        (24,  128, 15.0),   # 24 > 64? NO but 18 < 24 → was CPU, now GPU
    ])
    def test_no_regression(self, free_vram_gb, free_ram_gb, model_gb):
        """Fixed code never returns CPU when original returned GPU."""
        # Original logic: free_vram > free_ram/2 AND model*1.2 < free_vram
        mem_l = free_vram_gb
        mem_o = free_ram_gb
        if mem_l > (mem_o * 0.5) and model_gb * 1.2 < mem_l:
            old_result = "GPU"
        else:
            old_result = "CPU"

        with patch("comfy.model_management.get_free_memory",
                   return_value=int(free_vram_gb * 1024**3)):
            dev = text_encoder_initial_device(
                torch.device("xpu:0"), torch.device("cpu"),
                int(model_gb * 1024**3),
            )
            new_result = "GPU" if dev.type == "xpu" else "CPU"
            # If original said GPU, new must also say GPU
            if old_result == "GPU":
                assert new_result == "GPU", (
                    f"REGRESSION: free_vram={free_vram_gb} free_ram={free_ram_gb} "
                    f"model={model_gb}: original=GPU → fixed={new_result}"
                )

    # ── text_encoder_device integration ─────────────────────────────

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    @patch("comfy.model_management.should_use_fp16", return_value=True)
    @patch("comfy.model_management.get_torch_device", return_value=torch.device("xpu:0"))
    @patch("comfy.model_management.is_intel_xpu", return_value=True)
    def test_text_encoder_device_returns_xpu(self, *_):
        """text_encoder_device() returns xpu:0 with default NORMAL_VRAM."""
        dev = text_encoder_device()
        assert dev.type == "xpu"

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    @patch("comfy.model_management.should_use_fp16", return_value=True)
    @patch("comfy.model_management.get_torch_device", return_value=torch.device("xpu:0"))
    @patch("comfy.model_management.vram_state", type("VS", (), {"name": "LOW_VRAM"})())
    def test_text_encoder_device_low_vram(self, *_):
        """text_encoder_device() returns cpu with LOW_VRAM."""
        dev = text_encoder_device()
        assert dev.type == "cpu"

    @patch("comfy.model_management.get_free_memory", return_value=12.0 * 1024**3)
    @patch("comfy.model_management.get_torch_device", return_value=torch.device("xpu:0"))
    def test_text_encoder_device_gpu_only(self, *_):
        """--gpu-only forces text_encoder_device() to GPU."""
        with patch("comfy.model_management.args.gpu_only", True):
            dev = text_encoder_device()
            assert dev.type == "xpu"
