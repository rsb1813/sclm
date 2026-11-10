# Checks the SSD tier of the engine core: the unbuffered reader returns the file's bytes, and the cache policy decides
# as the LRFU replay in scripts/cache_policy.py does.
import math
import os

import numpy as np
import pytest

from sclm import core
from scripts import cache_policy

pytestmark = pytest.mark.skipif(not os.path.exists(core.DLL), reason="run engine\\build.bat first")


def test_reader_returns_the_bytes_of_any_range(tmp_path):
    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, n, dtype=np.uint8) for n in (3 * 2**20 + 1234, 70_000)]  # sizes off the page grid
    paths = [str(tmp_path / f"{i}.bin") for i in range(2)]
    for p, b in zip(paths, blobs):
        b.tofile(p)
    reader = core.Reader(paths, chunk=64 * 1024)
    ranges = [(0, 0, 10), (0, 4096, 8192), (0, 5000, 2**20 + 77), (1, 12345, 50_000),
              (0, len(blobs[0]) - 300_001, 300_001), (1, 0, len(blobs[1]))]  # the last two end at the end of the file
    for batch in (ranges[:1], ranges[1:]):  # one read alone, then several in flight at once
        bufs = [core.pages(n + 2 * core.PAGE) for _, _, n in batch]
        heads = [reader.submit(f, off, n, buf) for (f, off, n), buf in zip(batch, bufs)]
        reader.wait()
        for (f, off, n), buf, head in zip(batch, bufs, heads):
            assert head == off % core.PAGE
            assert np.array_equal(buf[head:head + n], blobs[f][off:off + n]), (f, off, n)


def test_reader_reports_a_missing_file(tmp_path):
    with pytest.raises(AssertionError):
        core.Reader([str(tmp_path / "absent.bin")])


def lrfu_calls(calls, n_keys, cap, lam):
    """The engine's rule in plain Python: cache_policy.lrfu, plus a slot used by a call is not evicted by that call."""
    score, slot_of, key_in, out = {}, {}, [], []
    for t, keys in calls:
        used, res = set(), []
        for i, k in enumerate(keys):
            a = lam * t
            if k in score:
                a = max(a, score[k]) + math.log1p(math.exp(-abs(a - score[k])))
            score[k] = a
            miss = k not in slot_of
            if miss:
                if len(key_in) < cap:
                    key_in.append(k)
                    slot_of[k] = len(key_in) - 1
                else:
                    free = [(score[kj], kj) for j, kj in enumerate(key_in) if j not in used]
                    if not free or a <= min(free)[0]:
                        res.append((cap + i, True))
                        continue
                    victim = min(free)[1]
                    slot_of[k] = slot_of.pop(victim)
                    key_in[slot_of[k]] = k
            used.add(slot_of[k])
            res.append((slot_of[k], miss))
        out.append(res)
    return out


def trace(n_keys, n_calls, width, seed):
    """Calls with topic-like locality: each draws distinct keys from a slowly moving window of the key space."""
    rng = np.random.default_rng(seed)
    calls = []
    for c in range(n_calls):
        lo = (c // 40 * 17) % (n_keys - 24)
        calls.append((c // 4, rng.choice(np.arange(lo, lo + 24), size=rng.integers(1, width + 1), replace=False).tolist()))
    return calls


@pytest.mark.parametrize("cap,lam", [(4, 0.3), (16, 0.1), (16, 0.0), (40, 2.0)])  # cap 4 < width: every slot can be in use
def test_cache_decides_as_the_python_rule(cap, lam):
    n_keys, calls = 96, trace(96, 3000, 6, seed=cap)
    cache = core.Cache(n_keys, cap, lam)
    want = lrfu_calls(calls, n_keys, cap, lam)
    for (t, keys), w in zip(calls, want):
        slot, miss = cache.use(keys, t)
        assert list(zip(slot.tolist(), miss.tolist())) == w, (t, keys)


def test_cache_misses_equal_the_replay_for_single_expert_calls():
    calls = trace(96, 4000, 1, seed=7)  # one key per call: the in-call rule never applies
    keys, toks = [k[0] for _, k in calls], [t for t, _ in calls]
    cache = core.Cache(96, 16, 0.1)
    misses = sum(int(cache.use([k], t)[1][0]) for k, t in zip(keys, toks))
    assert misses == cache_policy.lrfu(keys, toks, 16, 0.1)
    assert 0 < misses < len(keys)
