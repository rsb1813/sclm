# Puts the vendored upstream inference code (with the pure-PyTorch kernel.py) on sys.path for the tests.
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sclm", "reference"))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
