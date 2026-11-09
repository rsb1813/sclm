# Loads real checkpoint blocks through the layerwise runner (skipped until the needed shards are downloaded).
import os

import pytest
import torch

CKPT = os.environ.get("SCLM_CKPT", r"C:\models\DeepSeek-V4.1-Flash")


def have(*files: str) -> bool:
    return all(os.path.exists(os.path.join(CKPT, f)) for f in files)


needs_layer0 = pytest.mark.skipif(
    not have("model.safetensors.index.json", "tokenizer.json", "model-00002-of-00048.safetensors", "model-00003-of-00048.safetensors"),
    reason="layer-0 shards not downloaded",
)


@pytest.fixture(scope="module")
def runner():
    from transformers import AutoTokenizer

    from sclm.checkpoint import Checkpoint
    from sclm.layerwise import LayerwiseRunner, load_args

    torch.set_default_dtype(torch.bfloat16)
    args = load_args(max_batch_size=1, max_seq_len=1024)
    r = LayerwiseRunner(Checkpoint(CKPT), args, AutoTokenizer.from_pretrained(CKPT))
    yield r
    torch.set_default_dtype(torch.float32)


@needs_layer0
def test_layer0_loads_strictly_and_runs(runner):
    ids = torch.tensor([[0] + list(range(1000, 1255))])  # BOS then arbitrary ordinary tokens
    h = runner.hidden(ids, n_layers=1)
    assert h.shape == (1, 256, runner.args.dim)
    assert torch.isfinite(h).all()
    rms = h.float().pow(2).mean(-1).sqrt()
    assert (rms > 0).all() and (rms < 1e4).all(), rms.aminmax()
