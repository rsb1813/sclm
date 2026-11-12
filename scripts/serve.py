# An OpenAI-compatible HTTP server for the tiered model (DeepSeek-V4-Flash-0731): chat completions, streamed or whole.
# One thread computes. Requests that are open together take turns, a few tokens at a time, each in a token sequence of
# its own; a conversation that comes back with more messages finds its sequence still there and feeds only what is new.
import argparse
import contextlib
import io
import json
import os
import queue
import select
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import sclm.upstream  # noqa: F401  (puts the model's folder on sys.path)
from encoding_dsv4 import dsml_token, encode_messages, eos_token, parse_message_from_completion_text, tool_calls_block_name
from scripts.chat import looping, pick
from sclm import CKPT, CODES

CHUNK = 4096  # a sequence's room is a multiple of this many tokens
FEED = 16  # tokens of a prompt fed in one step: the decoder takes them through its layers together
RUN = 4096  # tokens of a long prompt fed in one step, as a run computed with the original experts (Host.prefill)
TOOL_CALLS = f"\n\n<{dsml_token}{tool_calls_block_name}"  # what a tool call block begins with in the answer's text
# OpenAI's reasoning_effort values as the model's (None: no thinking)
EFFORTS = {"none": None, "minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


class Refused(Exception):
    """A request the server does not take: (HTTP status, message)."""


def common(a: list, b: list) -> int:
    """How many leading tokens two lists share."""
    lo, hi = 0, min(len(a), len(b)) + 1  # a[:lo] == b[:lo]; a[:hi] != b[:hi] or hi is past the shorter one
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class Pieces:
    """The text of a growing list of tokens, a piece per token: a piece is held back while its last character is
    still missing bytes, and each is decoded with the piece before it, which decides how it begins."""

    def __init__(self, tok):
        self.tok, self.ids, self.a, self.b = tok, [], 0, 0

    def add(self, token: int) -> str:
        self.ids.append(token)
        before = self.tok.decode(self.ids[self.a:self.b], clean_up_tokenization_spaces=False)
        text = self.tok.decode(self.ids[self.a:], clean_up_tokenization_spaces=False)
        if text.endswith("�"):
            return ""
        self.a, self.b = self.b, len(self.ids)
        return text[len(before):]


class Conversation:
    """A token sequence the decoder holds (`seq`), the tokens fed to it, and `mark`: (position, the decoder's
    snapshot there) of the state before the last prompt token, where the sequence can be taken back to."""

    def __init__(self, seq):
        self.seq, self.ids, self.mark = seq, [], None


class Job:
    """One request: the prompt's tokens `ids`, then up to max_new tokens of reply, sampled. The engine calls step()
    in turn with the other jobs. What the job produces goes to `events`: ("think" | "answer", text), ("feed", tokens of
    the prompt fed, of how many), and last ("done", result) or ("error", status, message). stop: strings that end the
    answer; tools: the answer's tool calls are parsed out of it."""

    def __init__(self, ids: list, max_new: int, temperature: float = 1.0, top_p: float = 1.0, stop: tuple = (),
                 seed: int | None = None, tools: bool = False):
        self.ids, self.max_new, self.temperature, self.top_p, self.seed = ids, max_new, temperature, top_p, seed
        self.stops, self.tools = [s for s in stop if s], tools
        self.events, self.cancelled = queue.Queue(), False
        self.conv, self.at, self.reused = None, 0, 0  # set when the engine gives the job a sequence
        self.logits = self.rng = self.kind = self.pieces = None
        self.new, self.thought, self.said = 0, 0, []  # tokens of the reply, of its thinking, and the reply's tokens
        self.text = {"think": "", "answer": ""}
        self.shown, self.calling = 0, False  # how much of the answer was sent; a tool call has begun
        self.times = {"submit": time.perf_counter()}

    def _feed(self, engine, tokens: list) -> None:
        self.logits = engine.decoder.feed(tokens, len(self.conv.ids))
        self.conv.ids += tokens

    def step(self, engine) -> int:
        """Feeds up to FEED tokens of the prompt, or one sampled token. Returns how many; 0: the job is over."""
        if self.cancelled:
            return self._end("cancelled")
        self.times.setdefault("start", time.perf_counter())
        if self.at < len(self.ids):
            last = len(self.ids) - 1  # fed alone, from the mark
            if self.at == last:
                self.conv.mark = (self.at, engine.decoder.snapshot())
            if engine.long and last - self.at >= engine.long:  # a run takes the time of reading every expert once
                tokens = self.ids[self.at:min(last, self.at + RUN)]
                self.logits = engine.decoder.prefill(tokens, len(self.conv.ids))
                self.conv.ids += tokens
            else:
                tokens = self.ids[self.at:min(last, self.at + FEED)] or self.ids[last:]
                self._feed(engine, tokens)
            self.at += len(tokens)
            self.events.put(("feed", self.at, len(self.ids)))
            return len(tokens)
        if self.kind is None:  # the reply begins
            self.kind = "think" if self.ids[-1] == engine.think_start else "answer"
            self.pieces = Pieces(engine.tok)
            self.times["prompt"] = time.perf_counter()
            if self.seed is not None:
                self.rng = torch.Generator(device=self.logits.device).manual_seed(self.seed)
        if self.new >= self.max_new or not engine.fits(self.conv):
            return self._end("length")
        token = pick(self.logits, self.temperature, self.top_p, self.rng)
        if token == engine.eos:  # not fed: the next turn's prompt has it
            return self._end("stop")
        self._feed(engine, [token])
        self.new += 1
        self.said.append(token)
        self.times.setdefault("first", time.perf_counter())
        if token == engine.think_end and self.kind == "think":
            self.kind, self.pieces = "answer", Pieces(engine.tok)
        else:
            self.thought += self.kind == "think"
            with engine.lock:
                piece = self.pieces.add(token)
            if piece and not self._say(piece):
                return self._end("stop")
        return 1 if not looping(self.said) else self._end("length")  # a loop is cut off like a reply that is too long

    def _say(self, piece: str) -> bool:
        """Adds a piece to the text and sends what can be sent. False: a stop string ends the answer."""
        self.text[self.kind] += piece
        if self.kind == "think":
            self.events.put(("think", piece))
            return True
        text = self.text["answer"]
        if self.calling:  # inside a tool call block: it is not content, and is parsed at the end
            return True
        # nothing sent so far is part of one: what could be the start of a stop string or of a tool call block waits
        cut = min((i for i in (text.find(s, self.shown) for s in self.stops) if i >= 0), default=-1)
        call = text.find(TOOL_CALLS, self.shown) if self.tools else -1
        if cut >= 0 and (call < 0 or cut < call):
            self.text["answer"], end = text[:cut], cut
        elif call >= 0:
            self.calling, end, cut = True, call, -1
        else:
            marks = self.stops + ([TOOL_CALLS] if self.tools else [])
            end = len(text) - max((n for s in marks for n in range(1, len(s)) if text.endswith(s[:n])), default=0)
        if end > self.shown:
            self.events.put(("answer", text[self.shown:end]))
            self.shown = end
        return cut < 0

    def _end(self, reason: str) -> int:
        text, calls = self.text["answer"], []
        if self.calling:
            try:
                parsed = parse_message_from_completion_text(text + eos_token, "chat")
                text = parsed["content"]
                calls = [{"id": "call_" + uuid.uuid4().hex[:24], **call} for call in parsed["tool_calls"]]
            except (AssertionError, ValueError):  # a block cut off or malformed stays in the content
                pass
        if len(text) > self.shown and not calls:
            self.events.put(("answer", text[self.shown:]))
        self.times["end"] = time.perf_counter()
        self.events.put(("done", {
            "finish_reason": "tool_calls" if calls and reason == "stop" else reason, "content": text,
            "reasoning_content": self.text["think"], "tool_calls": calls,
            "usage": {"prompt_tokens": len(self.ids), "completion_tokens": self.new, "total_tokens": len(self.ids) + self.new,
                      "prompt_tokens_details": {"cached_tokens": self.reused},
                      "completion_tokens_details": {"reasoning_tokens": self.thought}}}))
        return 0


class Engine:
    """The thread that computes. context: tokens the open and the kept sequences can hold together; slice: tokens a
    job is given before the next job's turn (0: a job runs to its end); keep: sequences kept after their request for
    the conversation's next one. reserve: tokens the device memory set aside for sequences is for at first; it is
    doubled when they hold more, at the expert cache's cost, and brought down again when a quarter would do (None:
    the decoder's memory is left as it is). long: a prompt with at least this many tokens still to feed goes in as
    runs of up to RUN tokens (0: never), which no other request's turn interrupts."""

    def __init__(self, tok, decoder, context: int, slice: int = 64, keep: int = 8, reserve: int | None = None):
        self.tok, self.decoder, self.context, self.slice, self.keep = tok, decoder, context, slice, keep
        self.eos = tok.eos_token_id
        self.think_start, self.think_end = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
        self.inbox, self.lock = queue.Queue(), threading.Lock()  # the lock: the tokenizer is used by every thread
        self.active, self.waiting, self.idle = [], [], []  # jobs taking turns, jobs without room yet, conversations kept
        self.stopping, self.long = False, 512
        # the sequence in use while nothing is served, in place of the state the model was built with
        self.blank = decoder.sequence(128)
        decoder.switch(self.blank)
        self.base = self.reserve = reserve
        if reserve is not None:
            decoder.reserve(reserve)

    def submit(self, job: Job) -> None:
        self.inbox.put(job)

    def room(self) -> int:
        return self.context - sum(c.seq.capacity for c in [job.conv for job in self.active] + self.idle)

    def _drop(self, conv: Conversation) -> None:
        if conv.seq is self.decoder.seq:
            self.decoder.switch(self.blank)
        if conv in self.idle:
            self.idle.remove(conv)

    def _fit(self, extra: int, spare: Conversation | None = None) -> bool:
        """Room for `extra` more tokens: kept conversations other than `spare` go, the oldest first, and the device
        memory set aside follows what the sequences then hold. False: the open requests hold the rest."""
        while self.room() < extra:
            other = next((conv for conv in self.idle if conv is not spare), None)
            if other is None:
                return False
            self._drop(other)
        if self.reserve is not None:
            want = self.base
            while want < self.context - self.room() + extra:
                want *= 2
            want = min(want, self.context)
            if want > self.reserve or want * 4 <= self.reserve:
                print(f"{time.strftime('%H:%M:%S')} room for sequences: {self.reserve} -> {want} tokens", flush=True)
                self.decoder.reserve(want)
                self.reserve = want
        return True

    def fits(self, conv: Conversation) -> bool:
        """Whether the open conversation has room for one more token, a chunk more being made if not."""
        room = conv.seq.capacity
        if len(conv.ids) < room:
            return True
        extra = min(CHUNK, self.context - room)
        if extra <= 0 or not self._fit(extra):
            return False
        self.decoder.resize(conv.seq, room + extra)
        return True

    def _place(self, job: Job) -> bool:
        """Gives the job a sequence: the kept conversation that already holds the longest part of its prompt, taken
        back to its mark if the prompt parts from it after that, or a new one. It has room for the prompt; the reply
        makes more as it grows (fits). False: no room until a job ends."""
        ids, best, at = job.ids, None, 0
        for conv in self.idle:
            n = common(conv.ids, ids)
            if n == len(conv.ids) and n < len(ids):
                reach = n
            else:
                reach = conv.mark[0] if conv.mark is not None and conv.mark[0] <= min(n, len(ids) - 1) else 0
            if reach > at:
                best, at = conv, reach
        need = min(self.context, -(-(len(ids) + 1) // CHUNK) * CHUNK)
        extra = need - best.seq.capacity if best is not None else need
        if extra > 0 and not self._fit(extra, best):
            return False
        if best is None:
            best = Conversation(self.decoder.sequence(need))
        else:
            self.idle.remove(best)
            self.decoder.switch(best.seq)
            if at < len(best.ids):
                self.decoder.restore(best.mark[1])
                del best.ids[at:]
            if extra > 0:
                self.decoder.resize(best.seq, need)
        job.conv, job.at, job.reused = best, at, at
        return True

    def _start(self, job: Job) -> bool:
        if job.cancelled:
            return True
        try:
            placed = self._place(job)
        except torch.OutOfMemoryError:  # the device has less room than the budget counts on: nothing is kept, and
            for conv in list(self.idle):  # the job waits for the open ones, or is refused if there are none
                self._drop(conv)
            torch.cuda.empty_cache()
            placed = False
            if not self.active:
                job.events.put(("error", 503, "not enough device memory for this context"))
                return True
        if placed:
            self.active.append(job)
        return placed

    def _finish(self, job: Job, keep: bool = True) -> None:
        self.active.remove(job)
        if keep:
            used = min(self.context, -(-(len(job.conv.ids) + 1) // CHUNK) * CHUNK)
            if used < job.conv.seq.capacity:  # kept with the room it uses
                self.decoder.resize(job.conv.seq, used)
            self.idle.append(job.conv)
            while len(self.idle) > self.keep:
                self._drop(self.idle[0])
        else:
            self._drop(job.conv)
        t = job.times
        if "end" in t and "prompt" in t:
            print(f"{time.strftime('%H:%M:%S')} prompt {len(job.ids)} tokens ({job.reused} held) in {t['prompt'] - t['start']:.1f}s, "
                  f"reply {job.new} tokens at {job.new / max(t['end'] - t['prompt'], 1e-9):.1f} tok/s, waited "
                  f"{t['start'] - t['submit']:.1f}s; open {len(self.active)}, kept {len(self.idle)}", flush=True)

    def turn(self, wait: float = 0.0) -> None:
        """Takes the requests that have come (with nothing to compute, waits up to `wait` seconds for one) and gives
        every open job its slice."""
        try:
            while True:
                self.waiting.append(self.inbox.get(timeout=0.0 if self.active or self.waiting else wait))
        except queue.Empty:
            pass
        self.waiting = [job for job in self.waiting if not self._start(job)]
        for job in list(self.active):
            try:
                self.decoder.switch(job.conv.seq)
                left = self.slice or 1 << 62
                while left > 0:
                    fed = job.step(self)
                    if not fed:
                        self._finish(job)
                        break
                    left -= fed
            except Exception as e:  # the sequence's state is unknown: it is not kept
                job.events.put(("error", 500, f"{type(e).__name__}: {e}"))
                self._finish(job, keep=False)

    def run(self) -> None:
        while not self.stopping:
            self.turn(0.5)


def request_job(body: dict, engine: Engine, think: bool, effort: str, max_tokens: int) -> Job:
    """The job of a chat completion request (its JSON body), or Refused."""
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise Refused(400, "messages: a non-empty list is required")
    if body.get("n", 1) != 1:
        raise Refused(400, "n: only 1 is supported")
    if isinstance(body.get("thinking"), dict):  # DeepSeek's switch
        think = body["thinking"].get("type") != "disabled"
    if body.get("reasoning_effort") is not None:
        if body["reasoning_effort"] not in EFFORTS:
            raise Refused(400, f"reasoning_effort: one of {', '.join(EFFORTS)}")
        effort = EFFORTS[body["reasoning_effort"]]
        think, effort = think and effort is not None, effort or "low"
    messages = []
    for msg in body["messages"]:
        msg = dict(msg)
        if isinstance(msg.get("content"), list):  # content parts: the text ones
            msg["content"] = "".join(part.get("text", "") for part in msg["content"] if part.get("type") == "text")
        messages.append(msg)
    tools = body.get("tools") if body.get("tool_choice") != "none" else None
    if tools:  # upstream's encoding lists them in the system message
        if messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = tools
    try:
        text = encode_messages(messages, "thinking" if think else "chat", reasoning_effort=effort)
    except (AssertionError, NotImplementedError, KeyError, TypeError, AttributeError) as e:
        raise Refused(400, f"messages: {type(e).__name__}: {e}")
    with engine.lock:
        ids = engine.tok.encode(text, add_special_tokens=False)
    if len(ids) >= engine.context:
        raise Refused(400, f"the prompt has {len(ids)} tokens; the context holds {engine.context}")
    asked = body.get("max_completion_tokens") or body.get("max_tokens") or max_tokens
    stop = body.get("stop") or ()
    temperature, top_p = (1.0 if body.get(k) is None else float(body[k]) for k in ("temperature", "top_p"))
    return Job(ids, min(int(asked), max_tokens, engine.context - len(ids)), temperature, top_p,
               (stop,) if isinstance(stop, str) else tuple(stop), body.get("seed"), bool(tools))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # the engine prints a line per request
        pass

    def _json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        kind = "invalid_request_error" if status < 500 else "server_error"
        self._json(status, {"error": {"message": message, "type": kind, "param": None, "code": None}})

    def _gone(self) -> bool:
        """Whether the client has closed the connection."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def do_GET(self) -> None:
        app = self.server.app
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": app.model, "object": "model", "created": app.created, "owned_by": "sclm"}]})
        elif self.path == "/health":
            engine = app.engine
            self._json(200, {"status": "ok", "open": len(engine.active), "waiting": len(engine.waiting) + engine.inbox.qsize(),
                             "kept": len(engine.idle)})
        else:
            self._error(404, "unknown path")

    def do_POST(self) -> None:
        app = self.server.app
        if self.path != "/v1/chat/completions":
            return self._error(404, "unknown path")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            job = request_job(body, app.engine, app.think, app.effort, app.max_tokens)
        except Refused as e:
            return self._error(*e.args)
        except (ValueError, TypeError) as e:
            return self._error(400, f"the body is not a chat completion request: {e}")
        app.engine.submit(job)
        head = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": app.model}
        stream = bool(body.get("stream"))

        def chunk(delta: dict, finish: str | None = None) -> None:
            data = {**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                chunk({"role": "assistant", "content": ""})
            while True:
                try:
                    event = job.events.get(timeout=1.0)
                except queue.Empty:
                    if self._gone():
                        raise ConnectionError
                    continue
                if event[0] == "error":
                    if stream:
                        self.wfile.write(b"data: " + json.dumps({"error": {"message": event[2], "type": "server_error"}}).encode() + b"\n\n")
                        break
                    return self._error(event[1], event[2])
                if event[0] == "done":
                    result = event[1]
                    break
                if stream and event[0] == "feed":  # a long prompt: the connection is kept in use
                    self.wfile.write(f": prompt {event[1]}/{event[2]}\n\n".encode())
                    self.wfile.flush()
                elif stream and event[0] in ("think", "answer"):
                    chunk({"reasoning_content" if event[0] == "think" else "content": event[1]})
            if not stream:
                message = {"role": "assistant", "content": result["content"]}
                if result["reasoning_content"]:
                    message["reasoning_content"] = result["reasoning_content"]
                if result["tool_calls"]:
                    message["tool_calls"] = result["tool_calls"]
                return self._json(200, {**head, "object": "chat.completion", "usage": result["usage"],
                                        "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}]})
            if event[0] == "done":
                calls = [{"index": i, **call} for i, call in enumerate(result["tool_calls"])]
                chunk({"tool_calls": calls} if calls else {}, result["finish_reason"])
                if (body.get("stream_options") or {}).get("include_usage"):
                    data = {**head, "object": "chat.completion.chunk", "choices": [], "usage": result["usage"]}
                    self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (ConnectionError, OSError):  # the client left: its job ends at its next turn
            job.cancelled = True
            self.close_connection = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (this machine only by default)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--context", type=int, default=262144, help="tokens the sequences can hold together: a request's prompt "
                    "and reply at most")
    ap.add_argument("--reserve", type=int, default=32768, help="tokens the VRAM set aside for sequences is for at first "
                    "(6.7 MiB per thousand); it grows with what they hold, and the expert cache gives up slots for it")
    ap.add_argument("--max-tokens", type=int, default=20000, help="the most tokens of a reply, thinking included")
    ap.add_argument("--slice", type=int, default=64, help="tokens a request is given before the next one's turn (0: requests "
                    "are served one after the other)")
    ap.add_argument("--keep", type=int, default=8, help="conversations kept after their reply, for their next request")
    ap.add_argument("--long", type=int, default=512, help="a prompt with this many tokens to feed or more goes in as runs "
                    "of 4096 tokens computed with the original weights, about 30 s a run (0: never)")
    ap.add_argument("--effort", default="low", choices=["low", "high", "max"], help="thinking depth when a request names none")
    ap.add_argument("--no-think", action="store_true", help="no thinking unless a request asks for it")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=CODES, type=lambda v: None if v == "none" else v)
    ap.add_argument("--slots", type=int, default=-1)
    ap.add_argument("--ram-cache", type=int, default=-1)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-ram-margin", type=float, default=0.5)
    ap.add_argument("--wait-share", type=float, default=0.0, help="lossy speed option (0: exact), as the chat's /wait-share")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--guess", type=int, default=3)
    ap.add_argument("--touch", type=int, default=8)
    cli = ap.parse_args()
    # the model's own state is built small: the sequences are made as requests come
    cli.ram_share, cli.max_seq_len, cli.positions = 0.0, CHUNK, cli.context
    print("모델을 올리는 중입니다…", flush=True)
    import transformers

    from scripts.generate import load_runtime

    transformers.logging.set_verbosity_error()  # the tokenizer's note about a model type it does not know
    with contextlib.redirect_stdout(io.StringIO()):  # the loader's own progress lines
        tok, cache, decoder, info = load_runtime(cli, background=28)
    engine = Engine(tok, decoder, cli.context, cli.slice, cli.keep, min(cli.reserve, cli.context))
    engine.long = cli.long
    server = ThreadingHTTPServer((cli.host, cli.port), Handler)
    server.daemon_threads = True
    server.app = argparse.Namespace(engine=engine, model=os.path.basename(cli.ckpt), created=int(time.time()),
                                    think=not cli.no_think, effort=cli.effort, max_tokens=cli.max_tokens)
    worker = threading.Thread(target=engine.run, daemon=True)
    worker.start()
    print(f"sclm serve · {server.app.model} · http://{cli.host}:{cli.port}/v1 · 문맥 {cli.context} 토큰, 답 최대 {cli.max_tokens} 토큰 · "
          f"VRAM 슬롯 {cache.n_slots}개, RAM 캐시 {cli.ram_cache}개 · Ctrl+C로 끝냄", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    engine.stopping = True  # the token under way is finished: its reads are not left half done
    worker.join(timeout=10)


if __name__ == "__main__":
    main()
