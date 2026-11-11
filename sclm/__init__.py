# SCLM: local high-speed inference experiments for DeepSeek-V4.1-Flash and DeepSeek-V4-Flash-0731.
import os

# One model per process, picked by SCLM_MODEL: "v41" (DeepSeek-V4.1-Flash) or "v4" (DeepSeek-V4-Flash-0731).
MODEL = os.environ.get("SCLM_MODEL", "v41")
assert MODEL in ("v41", "v4"), MODEL
CKPT = r"C:\models\DeepSeek-V4.1-Flash" if MODEL == "v41" else r"C:\models\DeepSeek-V4-Flash-0731"
# layers, routed experts per layer, and an expert's input and hidden widths (the models' config.json)
N_LAYERS, N_EXPERTS, DIM, INTER = (40, 384, 5120, 2304) if MODEL == "v41" else (43, 256, 4096, 2048)
# activations per scale of the FP8 rounding before a quantized GEMM (upstream's linear)
ACT_BLOCK = 32 if MODEL == "v41" else 128
# the code set a runtime loads unless told otherwise (scripts: --codes, "none" for every expert original)
CODES = r"C:\models\sclm-codes\tier20v100" if MODEL == "v41" else r"C:\models\sclm-codes\v4-q175"
# the model's calibration sets and measurements: V4.1's in data/, the other model's in data/v4/
DATA = os.path.join(os.path.dirname(__file__), "..", "data", *([] if MODEL == "v41" else [MODEL]))
