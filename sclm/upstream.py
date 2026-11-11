# Imports the vendored upstream inference modules under their own top-level names. SCLM_MODEL picks the model:
# "v41" (DeepSeek-V4.1-Flash, sclm/reference) or "v4" (DeepSeek-V4-Flash-0731, sclm/reference_v4, which shares the
# pure-PyTorch kernel.py of sclm/reference). One model per process: both call their module `model`.
import os
import sys

from . import ACT_BLOCK, MODEL

_HERE = os.path.dirname(__file__)
_REF = os.path.join(_HERE, "reference")
if _REF not in sys.path:
    sys.path.insert(0, _REF)
if MODEL == "v4":
    _REF = os.path.join(_HERE, "reference_v4")
    sys.path.insert(0, _REF)  # ahead of sclm/reference: `model` is found here, `kernel` there

import model  # noqa: E402

engram = None  # V4.1 only
if MODEL == "v41":
    import engram  # noqa: E402

CONFIG_PATH = os.path.join(_REF, "config.json")
assert ACT_BLOCK == (model.block_size if MODEL == "v4" else model.fp8_block_size)
