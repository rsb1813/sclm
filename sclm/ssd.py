# SSD tier: a routed expert's original bytes (FP4 weights and their scales) read from the checkpoint into a staging buffer.
import numpy as np

from sclm import core
from sclm.checkpoint import Checkpoint


class ExpertReader:
    """An expert lies in its layer's shard as two runs: w1, w2, w3 scales back to back, and far from them the three
    packed-FP4 weights back to back. Both are read as they are, with the engine's unbuffered reader."""

    def __init__(self, ck: Checkpoint, chunk: int = 4 << 20):
        self.ck = ck
        paths = sorted({ck.path(k) for k in ck.weight_map if ".ffn.experts." in k})
        self.file = {p: i for i, p in enumerate(paths)}
        self.reader = core.Reader(paths, chunk)
        self.sizes = [n for _, _, n in self.extents(0, 0)]  # the same for every expert
        self.nbytes = sum(self.sizes) + 4 * core.PAGE

    def extents(self, layer: int, expert: int) -> list:
        """[(shard path, file offset, byte count)] of the expert's scales and of its weights."""
        prefix = f"layers.{layer}.ffn.experts.{expert}."
        return [self.ck.extent([f"{prefix}{w}.{part}" for w in ("w1", "w2", "w3")]) for part in ("scale", "weight")]

    def staging(self) -> np.ndarray:
        return core.pages(self.nbytes)

    def submit(self, layer: int, expert: int, buf: np.ndarray) -> tuple[int, int]:
        """Starts reading the expert into buf (from staging()). Returns where its scales and its weights begin in buf,
        valid once wait() returns."""
        at, starts = 0, []
        for (path, offset, nbytes), size in zip(self.extents(layer, expert), self.sizes):
            assert nbytes == size
            room = size + 2 * core.PAGE
            starts.append(at + self.reader.submit(self.file[path], offset, size, buf[at:at + room]))
            at += room
        return starts[0], starts[1]

    def wait(self) -> None:
        self.reader.wait()
