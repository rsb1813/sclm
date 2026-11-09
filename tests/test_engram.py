# Checks the engram hash layout against the released config using the real tokenizer (skipped if ref/ is absent).
import json
import os

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOK_DIR = os.path.join(ROOT, "ref", "deepseek_v41")
CONFIG = os.path.join(ROOT, "sclm", "reference", "config.json")

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(TOK_DIR, "tokenizer.json")), reason="tokenizer not downloaded")


@pytest.fixture(scope="module")
def hash_state():
    from transformers import AutoTokenizer

    import model as m
    from engram import EngramLayout, NgramHashState

    with open(CONFIG) as f:
        args = m.ModelArgs(**json.load(f))
    args.max_seq_len = 64
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    layout = EngramLayout.from_args(args)
    return args, layout, NgramHashState(args, layout, tok).cuda(), tok


def test_bucket_ranges_sum_to_table_rows(hash_state):
    args, layout, _, _ = hash_state
    for layer_primes, rows in zip(layout.primes, args.engram_num_embeddings):
        assert sum(p for per_ngram in layer_primes for p in per_ngram) == rows


def test_hash_ids_stay_in_their_bucket(hash_state):
    args, layout, state, tok = hash_state
    ids = torch.tensor([tok.encode("안녕하세요, DeepSeek 엔그램 해시 테스트입니다. The quick brown fox.")[:40]], device="cuda")
    hashes = state(ids, 0)  # [1, L, n_layers, (max_ngram-1) * n_heads]
    n_cols = (layout.max_ngram_size - 1) * layout.n_heads
    assert hashes.shape == (1, ids.size(1), len(layout.layer_ids), n_cols)
    primes = state.primes.flatten(1)  # [n_layers, n_cols]
    lo = state.offsets.cuda()
    assert ((hashes >= lo) & (hashes < lo + primes)).all()
    for li, rows in enumerate(args.engram_num_embeddings):
        assert hashes[..., li, :].max() < rows
