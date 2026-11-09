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


needs_layers0_3 = pytest.mark.skipif(
    not have("model-00004-of-00048.safetensors", "model-00005-of-00048.safetensors",
             "model-00006-of-00048.safetensors", "model-00047-of-00048.safetensors"),
    reason="layer 0-3 shards not downloaded",
)


@needs_layer0
@needs_layers0_3
def test_layer_major_batches_match_single_runs(runner):
    # 512 tokens exceed the 128 window, and layer 3 reads the compressed KV and top-k that layer 2 publishes,
    # so a wrong save/restore of the cross-layer state between batches shows up here
    g = torch.Generator().manual_seed(0)
    a, b = (torch.randint(1000, 100000, (1, 512), generator=g) for _ in range(2))
    both = runner.hidden_batches([a, b], n_layers=4)
    assert torch.equal(both[0], runner.hidden(a, n_layers=4).cpu())
    assert torch.equal(both[1], runner.hidden(b, n_layers=4).cpu())


@needs_layers0_3
def test_mapped_engram_matches_direct_gather():
    """Threaded, deduplicated row reads must equal a plain per-row gather, including repeated indices."""
    from sclm.checkpoint import Checkpoint, MappedEngramEmbedding

    ck = Checkpoint(CKPT)
    w, s = ck.memmap("layers.1.engram.embed.weight"), ck.memmap("layers.1.engram.embed.scale")
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, w.shape[0], (2, 300, 24), generator=g)
    idx[0, :10] = idx[1, :10]  # duplicates across positions
    got = MappedEngramEmbedding(w, s)(idx.cuda()).cpu()
    flat = idx.reshape(-1).numpy()
    wt = torch.from_numpy(w[flat]).view(torch.float8_e4m3fn).float()
    st = torch.from_numpy(s[flat]).view(torch.float8_e8m0fnu).float()
    want = (wt.unflatten(-1, (-1, 32)) * st.unsqueeze(-1)).flatten(-2).to(torch.bfloat16).view(*idx.shape, -1)
    assert torch.equal(got, want)


@needs_layer0
def test_layer0_loads_strictly_and_runs(runner):
    ids = torch.tensor([[0] + list(range(1000, 1255))])  # BOS then arbitrary ordinary tokens
    h = runner.hidden(ids, n_layers=1)
    assert h.shape == (1, 256, runner.args.dim)
    assert torch.isfinite(h).all()
    rms = h.float().pow(2).mean(-1).sqrt()
    assert (rms > 0).all() and (rms < 1e4).all(), rms.aminmax()
