# SCLM: local high-speed inference experiments for DeepSeek-V4.1-Flash and DeepSeek-V4-Flash-0731.
import os

# One model per process, picked by SCLM_MODEL: "v41" (DeepSeek-V4.1-Flash) or "v4" (DeepSeek-V4-Flash-0731).
MODEL = os.environ.get("SCLM_MODEL", "v41")
assert MODEL in ("v41", "v4"), MODEL
CKPT = r"C:\models\DeepSeek-V4.1-Flash" if MODEL == "v41" else r"C:\models\DeepSeek-V4-Flash-0731"
# the model's calibration sets and measurements: V4.1's in data/, the other model's in data/v4/
DATA = os.path.join(os.path.dirname(__file__), "..", "data", *([] if MODEL == "v41" else [MODEL]))
