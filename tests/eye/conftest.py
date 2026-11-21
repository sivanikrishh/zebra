import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make helpers.py importable


@pytest.fixture
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return True
