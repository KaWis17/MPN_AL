"""Shared fixtures. The suite runs offline: models are tiny and randomly initialised."""

import sys
from pathlib import Path

import pytest

# Works with or without `pip install -e .` (see scripts/_bootstrap.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bootstrap  # noqa: E402, F401

from mpn_al.training.synthetic import synthetic_cells  # noqa: E402, F401  (re-exported for the tests)

TINY_VIT = "timm:vit_tiny_patch16_224"  # 12 blocks, 192-d, patch 16, class token


@pytest.fixture(scope="session")
def tiny_model():
    """A random ViT-Tiny behind the base-model interface."""
    import torch

    from mpn_al.embeddings import build_base_model

    torch.manual_seed(0)
    return build_base_model(TINY_VIT, pretrained=False)


@pytest.fixture()
def cells_tile():
    return synthetic_cells()
