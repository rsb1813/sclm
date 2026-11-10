# Terminal dashboard of the project's state: the running job, system load, quality results, checklist and commits.
import ctypes
import glob
import json
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import time

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CODES = r"C:\models\sclm-codes"  # quant_eval --save-codes directories
N_LAYERS = 40
ROUTED_PARAMS = 543.6e9  # routed-expert weights (design.md)
KL_TARGET, TOP1_TARGET = 0.1, 0.9  # provisional targets (design.md 10)
GIB = 2**30
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW

LAYER_LINE = re.compile(r"gptq layer +(\d+): (\d+) calib tokens.*?peak ([\d.]+)GiB, RAM free ([\d.]+)GiB, total (\d+) min")


def parse_log(text: str) -> dict:
    """A quant_eval log's state: the layers of its last attempt as (layer, peak VRAM GiB, free RAM GiB, minutes),
    the final result when the run got there, and the last error line after the last layer."""
    layers, result, error = [], None, None
    for line in text.splitlines():
        m = LAYER_LINE.search(line)
        if m:
            layer = int(m[1])
            if layers and layer <= layers[-1][0]:  # a resumed attempt starts over from its checkpoint
                layers = []
            layers.append((layer, float(m[3]), float(m[4]), int(m[5])))
            result = error = None
        elif line.startswith('{"config"'):
            try:
                result = json.loads(line)
            except ValueError:
                pass
        elif "Error" in line:
            error = line.strip()
    return {"layers": layers, "result": result, "error": error}


def parse_checklist(text: str) -> tuple[list, list]:
    """checklist.md as (the plain bullets under "## 지금", [(section title, done, total)])."""
    now, sections, title = [], [], None
    for line in text.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            if title != "지금":
                sections.append([title, 0, 0])
        elif title == "지금" and line.startswith("- "):
            now.append(line[2:].strip())
        elif sections and re.match(r"\s*- \[[ x]\]", line):
            sections[-1][1] += "[x]" in line[:8]
            sections[-1][2] += 1
    return now, sections


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def run(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                              cwd=ROOT, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


_cache: dict = {}


def cached(key: str, seconds: float, fn):
    """fn()'s value, recomputed at most every `seconds` (process list and git are too slow for every frame)."""
    t, value = _cache.get(key, (0.0, None))
    if time.time() - t > seconds:
        value = fn()
        _cache[key] = (time.time(), value)
    return value


def eval_running() -> bool:
    out = run(["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\").CommandLine"])
    return "quant_eval" in out


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in ("total", "avail", "page_total", "page_avail", "virt_total",
                                                "virt_avail", "ext")]


def ram() -> tuple[float, float]:
    """(used, total) physical memory in GiB."""
    st = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return (st.total - st.avail) / GIB, st.total / GIB


def gpu() -> tuple[float, float, int, int] | None:
    """(used GiB, total GiB, utilization %, temperature C)."""
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
               "--format=csv,noheader,nounits"])
    try:
        used, total, util, temp = (float(v) for v in out.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return used / 1024, total / 1024, int(util), int(temp)


def bar(done: float, total: float, color: str, width: int | None = None) -> ProgressBar:
    return ProgressBar(total=total, completed=min(done, total), width=width, complete_style=color,
                       finished_style=color, style="grey23")


def ago(seconds: float) -> str:
    for unit, size in (("일", 86400), ("시간", 3600), ("분", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} 전"
    return "방금"


def job_panel() -> Panel:
    logs = sorted(glob.glob(os.path.join(DATA, "log_*.txt")), key=os.path.getmtime)
    if not logs:
        return Panel("기록 없음", title="작업", border_style="grey50")
    tag = os.path.basename(logs[-1])[4:-4]
    st = parse_log(read(logs[-1]))
    alive = cached("alive", 10, eval_running)
    layers, result = st["layers"], st["result"]
    done = layers[-1][0] + 1 if layers else 0
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    if result and not alive:
        state, color = "완료", "green"
    elif alive:
        state, color = ("● 실행 중" if int(time.time()) % 2 else "○ 실행 중"), "cyan"
    else:
        state, color = "멈춤", "red"
    grid.add_row("상태", Text(f"{state}  ·  기록 {ago(time.time() - os.path.getmtime(logs[-1]))}", style=color))
    if result and not alive:
        kl, top1 = result["kl_top64"], result["top1_agree"]
        grid.add_row("결과", Text.assemble((f"KL {kl:.4f}", "green" if kl <= KL_TARGET else "yellow"), "  ·  ",
                                         (f"top-1 {top1:.1%}", "green" if top1 >= TOP1_TARGET else "yellow"),
                                         f"  ·  ppl {result['ppl']:.3f}"))
    else:
        grid.add_row("층", bar(done, N_LAYERS, color))
        note = f"{done}/{N_LAYERS}"
        if layers:
            minutes = layers[-1][3]
            note += f"  ·  경과 {minutes}분"
            if alive and done < N_LAYERS:
                note += f"  ·  남은 시간 약 {max(1, round((N_LAYERS - done) * minutes / len(layers)))}분"
            elif alive:
                note += "  ·  마지막 평가 중"
        grid.add_row("", note)
        if layers:
            grid.add_row("최대", f"VRAM {max(p for _, p, _, _ in layers):.1f}GiB  ·  RAM 가용 최소 "
                                f"{min(r for _, _, r, _ in layers):.1f}GiB")
        if not alive:
            ckpt = os.path.exists(os.path.join(DATA, f"ckpt_{tag}.pt"))
            grid.add_row("", Text("체크포인트 있음 (이어하기 가능)" if ckpt else "체크포인트 없음", style="yellow"))
            if st["error"]:
                grid.add_row("오류", Text(st["error"], style="red", no_wrap=True, overflow="ellipsis"))
    return Panel(grid, title=f"작업  [bold]{tag}[/]", border_style=color, title_align="left")


def system_panel() -> Panel:
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(no_wrap=True, justify="right")
    g = gpu()
    if g:
        grid.add_row("VRAM", bar(g[0], g[1], "magenta"), f"{g[0]:.1f} / {g[1]:.0f}GiB")
        grid.add_row("GPU", bar(g[2], 100, "magenta"), f"{g[2]}%  {g[3]}°C")
    used, total = ram()
    grid.add_row("RAM", bar(used, total, "blue"), f"{used:.1f} / {total:.0f}GiB")
    disk = shutil.disk_usage("C:\\")
    grid.add_row("C:", bar(disk.used, disk.total, "yellow"), f"여유 {disk.free / GIB:.0f}GiB")
    for d in sorted(glob.glob(os.path.join(CODES, "*", ""))):
        files = glob.glob(os.path.join(d, "layer_*.safetensors"))
        size = sum(os.path.getsize(f) for f in files) / GIB
        grid.add_row("부호", bar(len(files), N_LAYERS, "green"),
                     f"{os.path.basename(d[:-1])} {size:.1f}GiB ({len(files)}/{N_LAYERS}층)")
    return Panel(grid, title="시스템", border_style="grey50", title_align="left")


def kl_bar(kl: float, width: int = 30, top: float = 0.16) -> Text:
    """A bar of KL on [0, top] with a mark at the target."""
    n, mark = min(width, round(kl / top * width)), round(KL_TARGET / top * width)
    color = "green" if kl <= KL_TARGET else "yellow"
    cells = [("━", color) if i < n else ("─", "grey30") for i in range(width)]
    cells[mark] = ("┃", "bold white")
    return Text.assemble(*cells)


def results_panel() -> Panel:
    table = Table(box=None, expand=True, padding=(0, 1), header_style="grey62")
    for name, justify in (("설정", "left"), ("RAM 계층", "right"), (f"KL (목표 ≤ {KL_TARGET})", "left"), ("", "right"),
                          ("top-1", "right"), ("ppl", "right"), ("SSD 속도 상한", "right")):
        table.add_column(name, justify=justify, no_wrap=True)
    for path in sorted(glob.glob(os.path.join(DATA, "eval_tier*.json")), key=os.path.getmtime):
        try:
            r = json.loads(read(path))
            kl, top1 = r["kl_top64"], r["top1_agree"]
        except (ValueError, KeyError):
            continue
        tier, speed = "", ""
        try:
            alloc = json.loads(read(os.path.join(DATA, f"alloc_{re.sub(r'_cal\d+$', '', r['config'])}.json")))
            tier = f"{alloc['ram_bpw']:.2f}bpw {alloc['ram_bpw'] * ROUTED_PARAMS / 8 / GIB:.1f}GiB"
            speed = f"{alloc['replay']['eval']['ssd_bound_tok_s']:.1f} tok/s"
        except (ValueError, KeyError):
            pass
        ok = kl <= KL_TARGET and top1 >= TOP1_TARGET
        table.add_row(Text(r["config"], style="bold" if ok else ""), tier, kl_bar(kl),
                      Text(f"{kl:.4f}", style="green" if kl <= KL_TARGET else "yellow"),
                      Text(f"{top1:.1%}", style="green" if top1 >= TOP1_TARGET else "yellow"), f"{r['ppl']:.3f}", speed)
    return Panel(table, title="품질  [grey62]원본 대비, 모델 전체 평가 (오래된 순)[/]", border_style="grey50",
                 title_align="left", subtitle="[grey62]속도는 설계 추정이며 실측한 tok/s는 아직 없음[/]",
                 subtitle_align="right")


def progress_panel() -> Panel:
    now, sections = parse_checklist(read(os.path.join(ROOT, "checklist.md")))
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(no_wrap=True, justify="right")
    for title, done, total in sections:
        grid.add_row(title, bar(done, total, "green" if done == total else "cyan"), f"{done}/{total}")
    lines = [Text("· " + line, no_wrap=True, overflow="ellipsis") for line in now]
    return Panel(Group(*lines, Text(""), grid) if lines else grid, title="진행", border_style="grey50",
                 title_align="left")


def commits_panel() -> Panel:
    out = cached("git", 30, lambda: run(["git", "log", "-12", "--format=%ct|%s"]))
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True, justify="right", style="grey62")
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for line in out.splitlines():
        t, _, subject = line.partition("|")
        grid.add_row(ago(time.time() - int(t)), subject)
    return Panel(grid, title="최근 커밋", border_style="grey50", title_align="left")


def render() -> Layout:
    layout = Layout()
    n_results = len(glob.glob(os.path.join(DATA, "eval_tier*.json")))
    layout.split_column(Layout(name="head", size=1), Layout(name="top", size=8),
                        Layout(results_panel(), size=n_results + 4), Layout(name="bottom"))
    head = Table.grid(expand=True)
    head.add_column()
    head.add_column(justify="right")
    head.add_row(Text(" SCLM  DeepSeek-V4.1-Flash 로컬 실행", style="bold cyan"),
                 Text(time.strftime("%H:%M:%S") + "  ·  q 종료 ", style="grey62"))
    layout["head"].update(head)
    layout["top"].split_row(Layout(job_panel(), ratio=1), Layout(system_panel(), ratio=1))
    layout["bottom"].split_row(Layout(progress_panel(), ratio=1), Layout(commits_panel(), ratio=1))
    return layout


def main() -> None:
    if "--once" in sys.argv:  # one frame, for a log or a quick look
        Console().print(render())
        return
    with Live(render(), screen=True, auto_refresh=False) as live:
        try:
            while True:
                live.update(render(), refresh=True)
                for _ in range(20):  # 2 s per frame
                    if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                        return
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
