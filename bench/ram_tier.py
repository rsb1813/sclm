# Benchmark of the CPU tier alone in the runtime's pattern: two real layers, the experts a recorded routing picks for
# them, random inputs, and unrelated work between the calls so that the pool's workers go to sleep as they do in a
# decode. Several builds of the engine core can be given: they take turns every few calls, so that the machine's other
# load hits them alike, and every output is compared bit for bit with the first build's.
import argparse
import os
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sclm import core  # noqa: E402

DIM, INTER, LIMIT = 5120, 2304, 10.0
PHASES = ("quant", "in tables", "w1 w3", "hidden", "w2", "sum", "waiting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlls", nargs="+", default=[core.DLL], help="builds of sclm_core.dll, the baseline first")
    ap.add_argument("--codes", default="C:/models/sclm-codes/tier20v100")
    ap.add_argument("--routing", default="H:/projects/sclm/data/routing_v1.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--calls", type=int, default=240, help="layer calls in a sweep")
    ap.add_argument("--sweeps", type=int, default=5, help="times every build runs every call")
    ap.add_argument("--block", type=int, default=8, help="calls a build runs before the next build's turn")
    ap.add_argument("--pauses", type=float, nargs="+", default=[0.0, 2.0], help="ms of unrelated work between calls")
    ap.add_argument("--host-us", type=float, nargs="+", default=[0.0],
                    help="us of the caller's own work between Layer.start and Layer.wait (0: one forward call)")
    cli = ap.parse_args()

    files = {}
    read_file = core.read_file
    core.read_file = lambda path: files[path] if path in files else files.setdefault(path, read_file(path))
    builds = []  # (name, lib, layers)
    for dll in cli.dlls:
        core.DLL = dll
        core.set_threads(cli.threads)
        layers = [core.Layer(os.path.join(cli.codes, f"layer_{i:02d}.safetensors"), DIM, INTER, LIMIT) for i in cli.layers]
        builds.append((os.path.basename(dll), layers[0].lib, layers))

    # the calls: (layer, experts, gates, x), from a recorded routing
    routing = np.load(cli.routing)
    rng = np.random.default_rng(0)
    layers = builds[0][2]
    size = [{e: sum(layer.arrays[f"{opt}.{w}.codes"][k].nbytes for w in ("w1", "w2", "w3"))
             for opt in {name.split(".")[0] for name in layer.arrays if name.endswith(".experts")}
             for k, e in enumerate(layer.arrays[f"{opt}.experts"].tolist())} for layer in layers]
    calls, recent, n_bytes = [], [[] for _ in layers], 0
    while len(calls) < cli.calls:
        # in a decode 39 other layers come between two calls of a layer, so nothing of its last call is left in the
        # CPU cache: here a call takes no expert of the layer's last two calls (which leaves it fewer experts than
        # a decode's 3.75)
        t, k = int(rng.integers(0, routing["indices"].shape[0])), len(calls) % len(layers)
        ids, w = routing["indices"][t, cli.layers[k]].tolist(), routing["weights"][t, cli.layers[k]].astype(np.float32)
        ram = [j for j, e in enumerate(ids) if e in layers[k].experts]
        if any(ids[j] in seen for j in ram for seen in recent[k]):
            continue
        recent[k] = recent[k][-1:] + [[ids[j] for j in ram]]
        calls.append((k, [ids[j] for j in ram], np.ascontiguousarray(w[ram]), (3 * rng.standard_normal(DIM)).astype(np.float32)))
        n_bytes += sum(size[k][ids[j]] for j in ram)
    print(f"{len(calls)} calls, {sum(len(c[1]) for c in calls) / len(calls):.2f} experts and "
          f"{n_bytes / len(calls) / 1e6:.1f} MB of codes a call, {cli.threads} threads")
    prepared = [[(layers[k].handle, (core.Expert * len(ids))(*[layers[k].experts[e] for e in ids]), g.ctypes.data, len(ids),
                  x.ctypes.data) for k, ids, g, x in calls] for _, _, layers in builds]

    def phases(lib):
        out = (core.C.c_double * 7)()
        lib.sclm_phase_seconds(out)
        return np.array(out)

    junk, y, reference = np.zeros(8192, np.float32), np.empty(DIM, np.float32), {}

    def busy(ms):
        end = perf_counter() + ms * 1e-3
        while perf_counter() < end:
            np.add(junk, 1.0, out=junk)

    print("us a call, the median over the sweeps. blocked: the caller inside the engine; elapsed: from the call's start "
          "to its result; then the phases the threads share as thread time over the thread count, and `waiting`, the "
          "rest of the call")
    print(f"{'build':22s} pause host blocked elapsed " + " ".join(f"{p:>9s}" for p in PHASES))
    for pause in cli.pauses:
        for host in cli.host_us:
            rows = np.zeros((len(builds), cli.sweeps, 9))  # blocked, elapsed, the engine's phases
            for sweep in range(cli.sweeps * len(builds)):
                for first in range(0, len(calls), cli.block):
                    b = (first // cli.block + sweep) % len(builds)
                    name, lib, _ = builds[b]
                    busy(max(pause, 0.3))  # the build before stops spinning
                    row, before = rows[b, sweep // len(builds)], phases(lib)
                    for i in range(first, min(first + cli.block, len(calls))):
                        handle, ex, g, count, x = prepared[b][i]
                        if i > first:
                            busy(pause)
                        t0 = perf_counter()
                        if host:
                            lib.sclm_layer_start(handle, ex, g, count, x)
                            t1 = perf_counter()
                            busy(host * 1e-3)
                            t2 = perf_counter()
                            lib.sclm_layer_wait(handle, y.ctypes.data)
                        else:
                            lib.sclm_layer_forward(handle, ex, g, count, x, y.ctypes.data)
                            t1 = t2 = t0
                        t3 = perf_counter()
                        row[0] += t3 - t2 + t1 - t0
                        row[1] += t3 - t0
                        assert np.array_equal(reference.setdefault(i, y.copy()), y), f"{name} differs on call {i}"
                    row[2:] += phases(lib) - before
            for b, (name, _, _) in enumerate(builds):
                m = np.median(rows[b], axis=0) / len(calls) * 1e6
                ph = m[2:]
                if ph[6]:
                    ph[:5] /= cli.threads
                    ph[6] -= ph[:5].sum()
                else:  # a build that times its phases one after the other by the clock
                    ph[6] = m[1] - ph[:6].sum()
                print(f"{name:22s} {pause:5.1f} {host:4.0f} {m[0]:7.0f} {m[1]:7.0f} " + " ".join(f"{p:9.1f}" for p in ph))
    print(f"the builds' outputs are bit-identical on all {len(reference)} calls")


if __name__ == "__main__":
    main()
