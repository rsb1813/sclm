# An ollama-style terminal chat with the tiered model: the conversation lives in the decoder's state, a reply is
# streamed token by token, and slash commands change the settings.
import argparse
import contextlib
import ctypes
import io
import os
import signal
import time

import torch
from rich.console import Console

from scripts import status
from scripts.generate import GIB, load_runtime

HELP = """[bold]명령[/]
  /?, /help              이 도움말
  /bye, /exit            끝내기 (Ctrl+D도 같음)
  /clear                 대화를 지우고 새 문맥에서 시작
  /show, /set            지금 설정과 문맥 사용량
  /set system <글>       시스템 프롬프트 (새 문맥부터 적용)
  /set think, /set nothink   생각 과정을 쓸지 (다음 답부터)
  /set effort <low|medium|high|max>   생각의 깊이 (새 문맥부터 적용)
  /set showthink, /set hidethink   생각 과정을 펼쳐 보일지, 한 줄로 접을지 (기본: 접음)
  /set temperature <수>  0이면 가장 확률 높은 토큰만 (같은 말을 반복하기 쉬움). 기본 0.7, 모델 공식값은 1.0
  /set top_p <수>        temperature가 0보다 클 때 쓰는 누적 확률 (기본 0.95)
  /set repeat_penalty <수>   최근 64토큰에 나온 토큰을 덜 고르게 하는 정도. 1이면 끔 (기본 1.1)
  /set num_predict <수>  답 하나의 최대 토큰 수 (기본 2048)
  /set wait-share <수>   속도 옵션: 0이면 정확, 0.25면 가벼운 SSD expert를 기다리지 않음 (기본 0.25)
  /set topk <수>         속도 옵션: 토큰마다 계산할 expert 수. 0이면 여섯 개 모두, 5면 조금 더 빠르고 4부터는 품질이 눈에 띄게 떨어짐
  /set verbose, /set quiet   답 뒤에 속도를 보일지
[bold]입력[/]
  \"\"\"로 시작하면 여러 줄을 쓸 수 있고, \"\"\"로 끝냅니다. 답이 나오는 중에 Ctrl+C를 누르면 멈춥니다."""


class ContextFull(Exception):
    """The turn does not fit in what is left of the sequence."""


def pick(logits: torch.Tensor, temperature: float, top_p: float, recent=(), penalty: float = 1.0) -> int:
    """The next token from logits [1, vocab]: the most likely one at temperature 0, else a sample from the smallest
    set of most likely tokens whose probability reaches top_p. penalty > 1 makes the tokens in `recent` less likely
    first: their logit is divided by it, or multiplied when negative."""
    if penalty != 1.0 and recent:
        logits, seen = logits.float().clone(), torch.tensor(sorted(set(recent)), device=logits.device)
        at = logits[0, seen]
        logits[0, seen] = torch.where(at > 0, at / penalty, at * penalty)
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits[0].float() / temperature, dim=-1)
    order = probs.argsort(descending=True)
    kept = probs[order]
    kept[kept.cumsum(0) - kept >= top_p] = 0
    return int(order[torch.multinomial(kept, 1)])


def looping(tokens: list, repeats: int = 6, span: int = 96) -> bool:
    """Whether a reply has run into a loop: its last `span` tokens or more are one piece of at most 32 tokens, said
    at least `repeats` times in a row. (A model that takes the likeliest token every time does not leave such a loop.)"""
    for period in range(1, 33):
        need = max(span, period * repeats)
        if len(tokens) >= need and tokens[-1] == tokens[-1 - period] and tokens[-need + period:] == tokens[-need:-period]:
            return True
    return False


REPEAT_LAST = 64  # tokens of the reply so far that the repetition penalty looks at


class Session:
    """One conversation as a single token sequence in the decoder: a turn is appended to it and nothing is fed
    twice. The model's thinking therefore stays in the context (the template's drop_thinking=false form)."""

    def __init__(self, tok, decoder, max_seq_len: int):
        self.tok, self.decoder, self.max_seq_len = tok, decoder, max_seq_len
        self.eos = tok.eos_token_id
        self.think_start, self.think_end = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
        self.system, self.think, self.effort = "", True, "high"
        self.clear()

    def clear(self) -> None:
        self.pos, self.logits = 0, None  # tokens fed so far, and the logits after the last of them

    def turn_ids(self, message: str) -> list:
        """The tokens a user message adds: the whole prompt at the start, later the end of the assistant's turn and
        what the template puts after it."""
        kw = {"thinking_mode": "thinking" if self.think else "chat", "reasoning_effort": self.effort}
        user = {"role": "user", "content": message}
        if self.pos == 0:
            messages = ([{"role": "system", "content": self.system}] if self.system else []) + [user]
            return self.tok.encode(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw))
        a, b = "\x02A\x02", "\x02B\x02"  # stand-ins for an earlier exchange: the template's text after it is wanted
        text = self.tok.apply_chat_template([{"role": "user", "content": a}, {"role": "assistant", "content": b}, user],
                                            add_generation_prompt=True, tokenize=False, **kw)
        return [self.eos] + self.tok.encode(text.split(b + self.tok.eos_token)[-1], add_special_tokens=False)

    def _feed(self, token: int) -> None:
        self.logits = self.decoder.logits(torch.tensor([[token]]), self.pos)
        self.pos += 1

    def ask(self, message: str, max_new: int, temperature: float, top_p: float, on_text, stop=lambda: False,
            repeat_penalty: float = 1.0) -> dict:
        """Feeds the user's message and generates the reply, handing each piece of text to on_text(kind, text) with
        kind "think" or "answer" (an empty text marks the change from one to the other). stop() is asked between
        tokens, and a reply that runs into a loop (looping) ends there. Returns the counts and times of the turn."""
        ids = self.turn_ids(message)
        if self.pos + len(ids) + 1 >= self.max_seq_len:
            raise ContextFull
        t0 = time.perf_counter()
        for token in ids:
            self._feed(token)
        t1 = time.perf_counter()
        kind, new, part, sent, said, looped = "think" if ids[-1] == self.think_start else "answer", 0, [], "", [], False
        while new < max_new and self.pos < self.max_seq_len and not stop():
            token = pick(self.logits, temperature, top_p, said[-REPEAT_LAST:], repeat_penalty)
            if token == self.eos:  # not fed: the next turn starts with it
                break
            self._feed(token)
            new += 1
            said.append(token)
            if looping(said):
                looped = True
                break
            if token == self.think_end and kind == "think":
                kind, part, sent = "answer", [], ""
                on_text(kind, "")
                continue
            part.append(token)
            text = self.tok.decode(part, clean_up_tokenization_spaces=False)
            if not text.endswith("\ufffd"):  # else a character is still missing bytes
                on_text(kind, text[len(sent):])
                sent = text
        return {"prompt_tokens": len(ids), "prompt_s": t1 - t0, "tokens": new, "seconds": time.perf_counter() - t1,
                "looped": looped}


def read_message(prompt: str) -> str:
    """One message from the terminal; a line starting with three double quotes opens a message of several lines."""
    line = input(prompt)
    if not line.startswith('"""'):
        return line
    lines = [line[3:]]
    while not lines[-1].endswith('"""'):
        lines.append(input("... "))
    lines[-1] = lines[-1][:-3]
    return "\n".join(lines)


def settings(session: Session, cache, opts: dict) -> str:
    """The settings in one line, as /show and the first screen print them."""
    return (f"생각 {'켬' if session.think else '끔'}({'펼침' if opts['showthink'] else '접음'}) · effort {session.effort} · "
            f"temperature {opts['temperature']:g} · top_p {opts['top_p']:g} · repeat_penalty {opts['repeat_penalty']:g} · "
            f"num_predict {opts['num_predict']} · wait-share {cache.wait_share:g} · topk {cache.topk}")


def command(line: str, session: Session, cache, opts: dict, console: Console) -> bool:
    """Runs a slash command; False ends the chat."""
    name, _, rest = line.strip().partition(" ")
    if name == "/set" and not rest.strip():
        name = "/show"
    if name in ("/bye", "/exit"):
        return False
    if name in ("/?", "/help"):
        console.print(HELP)
    elif name == "/clear":
        session.clear()
        console.print("[grey62]대화를 지웠습니다.[/]")
    elif name == "/show":
        console.print(f"[grey62]문맥 {session.pos} / {session.max_seq_len} 토큰 · {settings(session, cache, opts)}\n"
                      f"시스템 프롬프트: {session.system or '(없음)'}[/]", highlight=False)
    elif name == "/set":
        key, _, value = rest.strip().partition(" ")
        value = value.strip()
        try:
            if key == "system":
                session.system = value
                console.print("[grey62]시스템 프롬프트를 바꿨습니다" + ("." if session.pos == 0 else " (/clear 뒤부터 적용).") + "[/]")
            elif key in ("think", "nothink"):
                session.think = key == "think"
            elif key == "effort" and value in ("low", "medium", "high", "max"):
                session.effort = value
                if session.pos:
                    console.print("[grey62]/clear 뒤부터 적용됩니다.[/]")
            elif key in ("showthink", "hidethink"):
                opts["showthink"] = key == "showthink"
            elif key in ("temperature", "top_p", "repeat_penalty"):
                opts[key] = float(value)
            elif key == "num_predict":
                opts[key] = int(value)
            elif key == "wait-share":
                cache.wait_share = float(value)
            elif key == "topk":
                cache.topk = int(value)
            elif key in ("verbose", "quiet"):
                opts["verbose"] = key == "verbose"
            else:
                console.print("[yellow]모르는 설정입니다. /? 를 보세요.[/]")
        except ValueError:
            console.print("[yellow]값을 읽지 못했습니다.[/]")
    else:
        console.print("[yellow]모르는 명령입니다. /? 를 보세요.[/]")
    return True


def echo_typing(on: bool) -> None:
    """On a Windows console, whether keys typed are shown. Off while a reply streams, or they would land in the
    middle of it; turning it on again also throws away what was typed meanwhile, so it is not taken for the next
    message."""
    if os.name != "nt":
        return
    kernel32, mode = ctypes.windll.kernel32, ctypes.c_uint()
    stdin = kernel32.GetStdHandle(-10)
    if not kernel32.GetConsoleMode(stdin, ctypes.byref(mode)):  # input is not a console
        return
    if on:
        kernel32.FlushConsoleInputBuffer(stdin)
    kernel32.SetConsoleMode(stdin, mode.value | 4 if on else mode.value & ~4)  # ENABLE_ECHO_INPUT


def reply(line: str, session: Session, opts: dict, console: Console) -> None:
    """Streams the reply to one message, then a line with the speed. The thinking is streamed in grey, or folded
    into one line that counts its tokens (opts["showthink"])."""
    shown, stopped, thought = [None], [False], [0, time.perf_counter()]  # pieces of thinking so far, and since when

    def on_text(kind: str, text: str) -> None:
        if kind == "think" and not opts["showthink"]:
            shown[0], thought[0] = kind, thought[0] + 1
            if thought[0] % 8 == 1:  # rewritten in place
                console.print(f"생각 중… {thought[0]} 토큰", end="\r", style="grey50", highlight=False)
            return
        if kind != shown[0]:
            if kind == "think":
                console.print("생각 중…", style="grey50")
            elif shown[0] == "think" and opts["showthink"]:
                console.print("\n…생각 끝\n", style="grey50")
            elif shown[0] == "think":
                console.print(f"생각 끝 · {thought[0]} 토큰 · {time.perf_counter() - thought[1]:.0f}초      \n", style="grey50",
                              highlight=False)
            shown[0] = kind
        console.print(text, end="", style="grey50" if kind == "think" else None, markup=False, highlight=False, soft_wrap=True)

    # Ctrl+C only raises a flag that is read between tokens: a token cut half way would leave reads under way
    before = signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    echo_typing(False)
    try:
        stats = session.ask(line, opts["num_predict"], opts["temperature"], opts["top_p"], on_text, lambda: stopped[0],
                            opts["repeat_penalty"])
    except ContextFull:
        console.print("[yellow]문맥이 가득 찼습니다. /clear 로 새로 시작하세요.[/]")
        return
    finally:
        echo_typing(True)
        signal.signal(signal.SIGINT, before)
    console.print("\n")
    if stats["looped"]:
        console.print("[yellow]같은 구절이 되풀이되어 멈췄습니다. 되풀이된 글이 문맥에 남아 있으니 /clear 로 새로 시작하는 편이 "
                      "좋고, 자주 그러면 /set temperature 나 /set repeat_penalty 를 올려 보세요.[/]\n")
    if opts["verbose"]:
        note = " · 멈춤" if stopped[0] else " · 문맥 가득 참" if session.pos >= session.max_seq_len else ""
        console.print(f"[grey50]{stats['tokens']} 토큰 · {stats['tokens'] / max(stats['seconds'], 1e-9):.1f} tok/s · 입력 "
                      f"{stats['prompt_tokens']} 토큰 {stats['prompt_s']:.1f}초 · 문맥 {session.pos}/{session.max_seq_len}{note}[/]\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--codes", default=r"C:\models\sclm-codes\tier20p02",
                    help=r"the CPU tier's codes (C:\models\sclm-codes\tier20v100: slower, closer to the original)")
    ap.add_argument("--slots", type=int, default=320, help="original experts the VRAM cache keeps")
    ap.add_argument("--lam", type=float, default=0.1, help="cache score decay per token")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads of the RAM tier")
    ap.add_argument("--max-seq-len", type=int, default=4096, help="tokens a conversation can hold")
    ap.add_argument("--min-ram-margin", type=float, default=0.5, help="GiB of RAM that must stay free after loading")
    ap.add_argument("--wait-share", type=float, default=0.25, help="lossy speed option at the start (/set wait-share "
                    "changes it; 0: exact). 0.25 leaves the evaluation's KL where it is and is a quarter faster")
    ap.add_argument("--topk", type=int, default=0, help="lossy: routed experts computed per token (0: all six; /set topk)")
    ap.add_argument("--guess", type=int, default=4, help="experts of the next layer each layer guesses, to read those on "
                    "the SSD a layer early (0: none)")
    ap.add_argument("--touch", type=int, default=8, help="guessed experts of the next layer whose CPU-tier codes are read "
                    "into the CPU's cache while the device computes the attention before them (0: none)")
    ap.add_argument("--ram-cache", type=int, default=-1, help="experts kept in RAM after they leave the VRAM cache, "
                    "18.8 MB each of pinned memory (default: what fits, up to 1200, leaving 3 GiB free)")
    cli = ap.parse_args()
    cli.ram_share = 0.0
    if cli.ram_cache < 0:
        used, total = status.ram()
        codes = sum(os.path.getsize(os.path.join(cli.codes, f)) for f in os.listdir(cli.codes) if f.startswith("layer_")) / GIB
        cli.ram_cache = max(0, min(1200, int((total - used - codes - 3.0) * GIB / 18.8e6)))
    console = Console()
    console.print("[grey62]모델을 올리는 중입니다 (약 20초)…[/]")
    import transformers

    transformers.logging.set_verbosity_error()  # the tokenizer's note about a model type it does not know
    with contextlib.redirect_stdout(io.StringIO()):  # the loader's own progress lines
        tok, cache, decoder, info = load_runtime(cli, background=28)  # enough for read-ahead and for wait-share later
    # The model's card samples at temperature 1.0 and top_p 0.95. Taking the likeliest token every time (0, the
    # default until now, for runs that can be compared) let a reply fall into a loop it never left; 0.7 is a guess
    # between the two for a model whose distributions the coded experts have flattened a little.
    opts = {"temperature": 0.7, "top_p": 0.95, "repeat_penalty": 1.1, "num_predict": 2048, "verbose": True, "showthink": False}
    session = Session(tok, decoder, cli.max_seq_len)
    console.print(f"\n[bold]sclm chat[/] · {os.path.basename(cli.ckpt)} · VRAM {info['allocated_gib']:.1f}GiB, RAM 부호 "
                  f"{info['codes_gib']:.1f}GiB, RAM 캐시 {cli.ram_cache}개\n[grey62]{settings(session, cache, opts)}\n"
                  f"/? 도움말 · /set 설정 보기 · /bye 끝내기[/]\n", highlight=False)
    while True:
        try:
            line = read_message(">>> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            continue
        if line.startswith("/"):
            if not command(line, session, cache, opts, console):
                break
        elif line.strip():
            reply(line, session, opts, console)


if __name__ == "__main__":
    main()
