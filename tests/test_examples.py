"""Unit tests for Chapter 05 examples."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_device
from model_loading import run_model_loading_examples


def test_device_detection():
    """Test that device detection works."""
    device = get_device()
    assert device in ["cpu", "cuda", "mps"]


def test_model_loading_runs():
    """Test that model_loading examples run without errors."""
    # This is a basic smoke test
    try:
        run_model_loading_examples()
    except Exception as e:
        pytest.fail(f"model_loading examples failed: {e}")


def test_imports():
    """Test that all required modules can be imported."""
    import torch
    import transformers

    assert transformers.__version__
    assert torch.__version__
