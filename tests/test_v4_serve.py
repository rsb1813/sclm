# The server's engine and HTTP layer (scripts/serve.py) on a scripted decoder: what a conversation's sequence holds
# against upstream's encoding, requests taking turns, the room they share, the prompt cache on disk (what a request
# goes on from, what a folder holds, a restart, the budget), stop strings, tool calls and the responses.
# Run with SCLM_MODEL=v4 (needs the checkpoint's tokenizer).
import argparse
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from sclm import CKPT, MODEL

pytestmark = pytest.mark.skipif(MODEL != "v4" or not os.path.exists(os.path.join(CKPT, "tokenizer.json")), reason="set SCLM_MODEL=v4")


class Seq:
    def __init__(self, capacity: int):
        self.capacity, self.fed = capacity, []


class Decoder:
    """Sequences that record what they are fed. After a script's prompt a sequence answers with the script's reply,
    one token a call, then the end of the turn."""

    def __init__(self, vocab: int, eos: int):
        self.vocab, self.eos, self.scripts, self.order, self.seq = vocab, eos, [], [], Seq(0)
        self.reserved = []  # what the device memory for sequences was set to, in turn
        self.runs = []  # the lengths of the runs fed with prefill()
        self.made = []  # the sequences of the requests, in the order made

    def reserve(self, tokens):
        self.reserved.append(tokens)

    def sequence(self, capacity):
        self.made.append(Seq(capacity))
        return self.made[-1]

    def switch(self, seq):
        self.seq = seq

    def resize(self, seq, capacity):
        seq.capacity = capacity

    def clone(self, seq, capacity):
        new = self.sequence(capacity)
        new.fed = list(seq.fed)
        return new

    def rows(self, seq, start, end):
        return seq.fed[start:end]

    def put(self, seq, start, rows):  # what the sequence held from there on is the rows' now, as far as it has room
        assert start <= len(seq.fed)
        seq.fed[start:] = rows[:max(0, seq.capacity - start)]

    def snapshot(self, device=None):
        return len(self.seq.fed)

    def restore(self, kept):
        del self.seq.fed[kept:]

    def feed(self, tokens, pos):
        for i, token in enumerate(tokens):
            out = self.logits(torch.tensor([[token]]), pos + i)
        return out

    def prefill(self, tokens, pos, marks=()):
        self.runs.append(len(tokens))
        return self.feed(tokens, pos), {at: at for at in marks}

    def logits(self, ids, pos):
        fed = self.seq.fed
        assert pos == len(fed) < self.seq.capacity
        fed += ids.flatten().tolist()
        self.order.append(self.seq)
        out, token = torch.full((1, self.vocab), -1e9), self.eos
        for prompt, reply in self.scripts:
            n = len(fed) - len(prompt)
            if 0 <= n < len(reply) and fed[:len(prompt)] == prompt and fed[len(prompt):] == reply[:n]:
                token = reply[n]
        out[0, token] = 0.0
        return out


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def made(tok, folder, context=1 << 16, **kw):
    """An engine with a decoder of its own and the prompt cache in `folder`. Its first sequence is the engine's blank."""
    from scripts.serve import Engine

    decoder = Decoder(len(tok), tok.eos_token_id)
    engine = Engine(tok, decoder, context, str(folder), **kw)
    del decoder.made[:]
    return engine, decoder


def folders(engine) -> int:
    return len(os.listdir(engine.cache))


def serve(engine, *jobs) -> list:
    """Runs the jobs to their end: each one's (streamed thinking, streamed answer, result)."""
    for job in jobs:
        engine.submit(job)
    while engine.active or engine.waiting or not engine.inbox.empty():
        engine.turn()
    out = []
    for job in jobs:
        text, result = {"think": "", "answer": ""}, None
        while not job.events.empty():
            event = job.events.get()
            if event[0] in text:
                text[event[0]] += event[1]
            elif event[0] != "feed":
                result = event
        assert result[0] == "done", result
        out.append((text["think"], text["answer"], result[1]))
    return out


@pytest.mark.parametrize("think", [True, False])
def test_a_conversation_goes_on_from_what_it_held_and_holds_what_upstream_encodes(tok, tmp_path, think):
    import sclm.upstream  # noqa: F401
    from encoding_dsv4 import encode_messages
    from scripts.serve import request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok, tmp_path)
    think_end = tok.convert_tokens_to_ids("</think>")
    turns = [("Rust로 숫자를 양자화하는 코드를 써 줘.", "Let me think about rounding.", "fn q(x: f64) -> i32 { x.round() as i32 }"),
             ("고마워, 테스트도 추가해 줘.", "A unit test with assert_eq.", "#[test] fn t() { assert_eq!(q(1.6), 2); }"),
             ("끝.", "Nothing more to do.", "네.")]
    sent, full, held = [{"role": "system", "content": "You answer briefly."}], [], 0
    for question, reasoning, answer in turns:
        sent.append({"role": "user", "content": question})  # as a client sends it: the earlier answers without their thinking
        job = request_job({"messages": sent, "temperature": 0, "thinking": {"type": "enabled" if think else "disabled"},
                           "reasoning_effort": "high"}, engine, True, "low", 500)
        reply = (enc(reasoning) + [think_end] if think else []) + enc(answer)
        decoder.scripts.append((list(job.ids), reply))
        thought, said, result = serve(engine, job)[0]
        assert said == result["content"] == tok.decode(enc(answer)) and result["finish_reason"] == "stop"
        assert thought == result["reasoning_content"] == (tok.decode(enc(reasoning)) if think else "")
        usage = result["usage"]
        assert usage["completion_tokens"] == len(reply) and usage["prompt_tokens"] == len(job.ids)
        assert usage["completion_tokens_details"]["reasoning_tokens"] == (len(enc(reasoning)) if think else 0)
        # everything the sequence held is used again: all of it, or up to where the last reply's thinking began
        assert usage["prompt_tokens_details"]["cached_tokens"] == held
        held = len(job.ids) - 1 if think else len(job.ids) + len(reply)
        sent.append({"role": "assistant", "content": answer})
        full += [sent[-2], {"role": "assistant", "content": answer, **({"reasoning_content": reasoning} if think else {})}]
    want = enc(encode_messages(sent[:1] + full, "thinking" if think else "chat", reasoning_effort="high"))
    assert len(engine.idle) == 1 and engine.idle[0].ids == want[:-1]  # the last end-of-turn token comes with the next turn
    # each turn's sequence was read from the turn before's folder, which the turn's own then replaced
    assert len(decoder.made) == 3 and decoder.made[-1].fed == want[:-1] and folders(engine) == 1
    assert engine.idle[0].seq is None and engine.room() == engine.context  # a kept conversation holds no memory


def test_requests_open_together_take_turns_and_share_the_room(tok, tmp_path):
    from scripts.serve import CHUNK, Job

    engine, decoder = made(tok, tmp_path, context=2 * CHUNK, slice=2)
    jobs = []
    for n in range(3):  # each needs a chunk of room: two are open together, the third waits for one of them to end
        prompt, reply = [10 + n] * (7 + n), [20 + n] * (6 + 2 * n)
        decoder.scripts.append((prompt, reply))
        jobs.append(Job(prompt, 100, temperature=0.0))
    results = serve(engine, *jobs)
    first, second, third = decoder.made
    for n, (job, seq, (_, _, result)) in enumerate(zip(jobs, decoder.made, results)):
        assert seq.fed == job.conv.ids == [10 + n] * (7 + n) + [20 + n] * (6 + 2 * n) and result["finish_reason"] == "stop"
    order = decoder.order
    # a prompt goes in together up to its last token, whatever the slice; from there the turns are two tokens long
    assert order[:21] == [first] * 6 + [second] * 7 + [first, first, second, second] * 2
    assert first not in order[order.index(third):]  # the third began when the first ended and left its room
    assert engine.room() == 2 * CHUNK and len(engine.idle) == 3 == folders(engine)  # all three are kept, on disk


def test_a_reply_makes_room_as_it_grows_up_to_the_context(tok, tmp_path):
    import random

    from scripts.serve import CHUNK, Job

    engine, decoder = made(tok, tmp_path, context=4 * CHUNK, cache_gb=0, reserve=CHUNK)
    words = random.Random(0).choices(range(1000, 2000), k=5 * CHUNK)  # no loop in them
    decoder.scripts += [([5, 6, 7], words[:2 * CHUNK + 9]), ([8, 9], words), ([3], words[:4])]
    grown, full, small = (Job(prompt, 10 * CHUNK, temperature=0.0) for prompt, _ in decoder.scripts)
    (_, _, result), = serve(engine, grown)
    assert result["finish_reason"] == "stop" and result["usage"]["completion_tokens"] == 2 * CHUNK + 9
    assert decoder.reserved == [CHUNK, 2 * CHUNK, 4 * CHUNK]  # doubled as the sequence passed a chunk, then two
    (_, _, result), = serve(engine, full)
    assert result["finish_reason"] == "length" and result["usage"]["total_tokens"] == 4 * CHUNK == len(decoder.made[1].fed)
    serve(engine, small)
    # each request's memory follows what it holds, from the least again
    assert decoder.reserved == [CHUNK, 2 * CHUNK, 4 * CHUNK] * 2 + [CHUNK] and engine.room() == 4 * CHUNK
    assert not engine.idle and folders(engine) == 0  # with no room on disk nothing is kept


def test_a_long_prompt_goes_in_as_runs_up_to_its_last_token(tok, tmp_path):
    from scripts.serve import RUN, Job

    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    long, short = [11] * (RUN + 700) + [12], [13] * 300 + [14]
    decoder.scripts += [(long, [21, 22]), (short, [23])]
    (_, _, first), (_, _, second) = serve(engine, Job(long, 10, temperature=0.0), Job(short, 10, temperature=0.0))
    assert decoder.runs == [RUN, 700] and first["usage"]["completion_tokens"] == 2 and second["usage"]["completion_tokens"] == 1
    assert decoder.made[0].fed == engine.idle[0].ids == long + [21, 22]
    # the ends of the runs (the second: before the last prompt token) and the end of the reply, which has no thinking
    assert sorted(engine.idle[0].marks) == [RUN, RUN + 700, RUN + 703]


def test_another_question_on_the_same_document_goes_on_from_the_documents_last_run(tok, tmp_path):
    from scripts.serve import PIECE, RUN, Job

    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    document = [11] * (2 * RUN + 700)
    first, second = document + [12] * 5 + [13], document + [14] * 9 + [15]
    decoder.scripts += [(first, [21, 22]), (second, [23])]
    (_, _, one), = serve(engine, Job(first, 10, temperature=0.0))
    assert decoder.runs == [RUN, RUN, 705] and one["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    (_, _, two), = serve(engine, Job(second, 10, temperature=0.0))
    # the two prompts part inside the third run: the sequence goes back to the end of the second
    assert two["usage"]["prompt_tokens_details"]["cached_tokens"] == 2 * RUN and decoder.runs[3:] == [709]
    # the first question's conversation had little after that mark: the second one's takes its place
    assert len(engine.idle) == 1 == folders(engine) and decoder.made[1].fed == engine.idle[0].ids == second + [23]
    assert sorted(engine.idle[0].marks) == [RUN, 2 * RUN, 2 * RUN + 709, 2 * RUN + 711]
    # its rows: the first question's file up to the mark, as a link, and its own from there
    assert PIECE == 2 * RUN and [piece[:2] for piece in engine.idle[0].chunks] == [[0, PIECE], [PIECE, 2 * RUN + 711]]
    assert sorted(os.stat(os.path.join(engine.idle[0].home, name)).st_nlink for _, _, name in engine.idle[0].chunks) == [1, 1]


def test_the_prompt_cache_outlives_the_server(tok, tmp_path):
    from scripts.serve import RUN, Job

    document = [11] * (2 * RUN + 700)
    first, second, third = document + [12] * 5 + [13], document + [14] * 9 + [15], document + [14] * 9 + [15, 23, 16, 17]
    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    decoder.scripts.append((first, [21, 22]))
    serve(engine, Job(first, 10, temperature=0.0))
    kept = os.listdir(engine.cache)
    # what a server leaves behind: the folder of a request that did not end, and one that is not a conversation
    for name, files in (("open", {"mark_5.pt": b"x"}), ("other", {"meta.json": b"{}"})):
        os.mkdir(tmp_path / name)
        for file, data in files.items():
            (tmp_path / name / file).write_bytes(data)
    engine, decoder = made(tok, tmp_path, context=4 * RUN)  # another server on the same folder
    assert os.listdir(engine.cache) == kept and [conv.ids for conv in engine.idle] == [first + [21, 22]]
    decoder.scripts += [(second, [23]), (third, [24])]
    (_, _, two), = serve(engine, Job(second, 10, temperature=0.0))
    assert two["usage"]["prompt_tokens_details"]["cached_tokens"] == 2 * RUN and decoder.runs == [709]
    assert decoder.made[0].fed == second + [23] and len(engine.idle) == 1 == folders(engine)
    # and once more, from the end of that reply: rows written by the first server and by the second
    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    decoder.scripts.append((third, [24]))
    (_, _, three), = serve(engine, Job(third, 10, temperature=0.0))
    assert three["usage"]["prompt_tokens_details"]["cached_tokens"] == len(second) + 1 and decoder.runs == []
    assert decoder.made[0].fed == third + [24] and len(engine.idle) == 1 == folders(engine)


def test_a_folder_that_cannot_be_read_is_no_cache(tok, tmp_path):
    from scripts.serve import RUN, Job

    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    document = [11] * (RUN + 700)
    first, second = document + [12] * 5 + [13], document + [14] * 9 + [15]
    decoder.scripts += [(first, [21, 22]), (second, [23])]
    serve(engine, Job(first, 10, temperature=0.0))
    bad = engine.idle[0]
    with open(os.path.join(bad.home, f"mark_{RUN}.pt"), "wb") as f:
        f.write(b"not a snapshot")
    (_, _, two), = serve(engine, Job(second, 10, temperature=0.0))  # answered all the same, from the start
    assert two["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and two["usage"]["completion_tokens"] == 1
    assert [conv.ids for conv in engine.idle] == [second + [23]] and os.listdir(engine.cache) == [os.path.basename(engine.idle[0].home)]
    # a folder deleted while the server runs
    import shutil

    shutil.rmtree(engine.idle[0].home)
    (_, _, again), = serve(engine, Job(first, 10, temperature=0.0))
    assert again["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and again["usage"]["completion_tokens"] == 2
    assert [conv.ids for conv in engine.idle] == [first + [21, 22]] and folders(engine) == 1


def test_a_disk_that_cannot_be_written_keeps_nothing(tok, tmp_path, monkeypatch):
    from scripts.serve import RUN, Job

    engine, decoder = made(tok, tmp_path, context=4 * RUN)
    prompt = [11] * (RUN + 700) + [12]
    decoder.scripts.append((prompt, [21, 22]))

    def full(what, path):
        open(path, "wb").close()  # as a write that fails half way leaves it
        raise RuntimeError("PytorchStreamWriter failed writing file")

    monkeypatch.setattr(torch, "save", full)
    (_, _, result), = serve(engine, Job(prompt, 10, temperature=0.0))  # answered all the same
    assert result["finish_reason"] == "stop" and result["usage"]["completion_tokens"] == 2
    assert not engine.idle and folders(engine) == 0 and not engine.active
    monkeypatch.undo()
    (_, _, result), = serve(engine, Job(prompt, 10, temperature=0.0))
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and len(engine.idle) == 1 == folders(engine)


def test_the_kept_conversations_stay_within_their_room_on_disk(tok, tmp_path):
    from scripts.serve import Job

    engine, decoder = made(tok, tmp_path, context=1 << 16)
    system = [31] * 900
    prompts = [system + [40 + k] * 2500 + [50 + k] for k in range(4)]
    decoder.scripts += [(prompt, [60 + k]) for k, prompt in enumerate(prompts)]
    session = lambda k: serve(engine, Job(prompts[k], 10, temperature=0.0, marks=[900], pins=[900]))[0][2]  # noqa: E731
    sizes = lambda conv: {name: os.path.getsize(os.path.join(conv.home, name)) for name in os.listdir(conv.home)}  # noqa: E731
    session(0)
    one, = engine.idle
    assert engine.used() == sum(sizes(one).values())
    session(1)
    two = engine.idle[1]
    # the second session's folder links to the first one's rows and to its mark after the system prompt: counted once
    linked = {name: size for name, size in sizes(two).items() if os.stat(os.path.join(two.home, name)).st_nlink > 1}
    assert sorted(name.split("_")[0] for name in linked) == ["mark", "rows"] and "mark_900.pt" in linked
    assert engine.used() == sum(sizes(one).values()) + sum(sizes(two).values()) - sum(linked.values())
    session(2)
    three = engine.idle[2]
    assert engine.idle == [two, one, three]  # the third began from the first one's mark: the first was used after the second
    # room for what a fourth session writes less a session's own files: the one unused for the longest goes
    own = next(size for name, size in sizes(two).items() if name.startswith("rows") and name not in linked)
    engine.budget = engine.used() + own + 2000
    assert session(3)["usage"]["prompt_tokens_details"]["cached_tokens"] == 900  # from the second one's mark
    assert engine.idle[:2] == [three, two] and len(engine.idle) == 3 == folders(engine) and not os.path.exists(one.home)
    assert engine.used() <= engine.budget
    # the first one's rows, which the others link to, are still there: the second session goes on from its reply's end
    engine.budget *= 2
    more = prompts[1] + [61, 70, 71]
    decoder.scripts.append((more, [72]))
    (_, _, result), = serve(engine, Job(more, 10, temperature=0.0, marks=[900], pins=[900]))
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == len(prompts[1]) + 1 and decoder.made[-1].fed == more + [72]
    assert len(engine.idle) == 3 == folders(engine) and not os.path.exists(two.home)  # in a folder that replaces the second's


def test_sessions_with_the_same_system_prompt_each_go_on_from_a_copy_of_it(tok, tmp_path):
    from scripts.serve import RUN, TAIL, Job

    engine, decoder = made(tok, tmp_path, context=8 * RUN)
    system = [31] * 900
    one, two, three = (system + [40 + k] * n + [50 + k] for k, n in enumerate((TAIL + 500, 60, 700)))
    decoder.scripts += [(one, [61]), (two, [62]), (three, [63])]
    serve(engine, Job(one, 10, temperature=0.0, marks=[900], pins=[900]))
    first = engine.idle[0]
    # the second session parts from the first after the system prompt, where the first has a long way behind it
    (_, _, result), = serve(engine, Job(two, 10, temperature=0.0, marks=[900], pins=[900]))
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert engine.idle[0] is first and first.ids == one + [61] and len(engine.idle) == 2 == folders(engine)
    assert decoder.made[1].fed == engine.idle[1].ids == two + [62]
    # two requests that come together: the second waits for the first to feed what they share, then copies it
    again = system + [47] * (RUN + 600)
    a, b = again + [48] * 30 + [58], again + [49] * 30 + [59]
    decoder.scripts += [(a, [64]), (b, [65])]
    before = len(decoder.runs)
    (_, _, ra), (_, _, rb) = serve(engine, Job(a, 10, temperature=0.0, marks=[900], pins=[900]),
                                   Job(b, 10, temperature=0.0, marks=[900], pins=[900]))
    assert ra["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert rb["usage"]["prompt_tokens_details"]["cached_tokens"] == 900 + RUN  # the end of the first one's first run
    assert decoder.runs[before:] == [RUN, 630, 630] and first.ids == one + [61]
    assert decoder.made[2].fed == a + [64] and decoder.made[3].fed == b + [65] and len(engine.idle) == 4 == folders(engine)
    # the copy was of an open request, whose rows were not on disk yet: its folder has them all, and a server reads it
    engine, decoder = made(tok, tmp_path, context=8 * RUN)
    more = b + [65, 66, 67]
    decoder.scripts.append((more, [68]))
    (_, _, result), = serve(engine, Job(more, 10, temperature=0.0, marks=[900], pins=[900]))
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == len(b) + 1 and decoder.made[0].fed == more + [68]


def test_a_request_can_ask_for_a_mark_after_a_part_of_a_message(tok, tmp_path):
    from scripts.serve import request_job

    engine, _ = made(tok, tmp_path)
    document = "def area(w, h):\n    return w * h\n" * 20
    body = {"messages": [{"role": "system", "content": "You review code, one finding a line. " * 8},
                         {"role": "user", "content": [{"type": "text", "text": document, "cache_control": {"type": "ephemeral"}},
                                                      {"type": "text", "text": "이 함수의 문제를 한 줄로 말해 줘."}]}], "temperature": 0}
    job = request_job(body, engine, True, "low", 500)
    plain = request_job({**body, "messages": [body["messages"][0], {"role": "user", "content": document + "이 함수의 문제를 한 줄로 말해 줘."}]},
                        engine, True, "low", 500)
    assert job.ids == plain.ids  # the same prompt as without the request for a mark
    after, turn = sorted(job.pins, reverse=True)  # after the document, and where the user's turn begins
    assert tok.decode(job.ids[:after]).endswith(document) and tok.decode(job.ids[after:]).startswith("이 함수")
    assert job.ids[turn] == engine.user and job.marks == sorted(job.pins)


def test_stop_strings_and_the_token_limit_end_a_reply(tok, tmp_path):
    from scripts.serve import Job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok, tmp_path)
    reply = enc("alpha beta gamma delta epsilon")
    decoder.scripts += [([5, 6], reply), ([7, 8], reply)]
    stopped, cut = serve(engine, Job([5, 6], 100, temperature=0.0, stop=("gam", " delta eps")),
                         Job([7, 8], 3, temperature=0.0, stop=("zeta",)))
    assert stopped[1] == stopped[2]["content"] == "alpha beta " and stopped[2]["finish_reason"] == "stop"
    assert cut[2]["finish_reason"] == "length" and cut[2]["usage"]["completion_tokens"] == 3
    assert cut[1] == cut[2]["content"] == tok.decode(reply[:3])


@pytest.mark.parametrize("think", [False, True])
def test_a_tool_call_is_parsed_out_of_the_answer(tok, tmp_path, think):
    import sclm.upstream  # noqa: F401
    from encoding_dsv4 import encode_messages
    from scripts.serve import request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok, tmp_path)
    mode, switch = ("thinking", {"type": "enabled"}) if think else ("chat", {"type": "disabled"})
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
    messages = [{"role": "user", "content": "서울 날씨 알려 줘."}]
    call = {"type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Seoul", "days": 2})}}
    said = {"role": "assistant", "content": "확인해 보겠습니다.", "tool_calls": [call],
            **({"reasoning_content": "The weather tool gives this."} if think else {})}
    # upstream's own text for the assistant's turn: what follows the prompt in the encoding of the conversation with it
    system = {"role": "system", "content": "", "tools": tools}
    prompt = encode_messages([system] + messages, mode)
    text = encode_messages([system] + messages + [said], mode)
    assert text.startswith(prompt)
    job = request_job({"messages": messages, "tools": tools, "temperature": 0, "thinking": switch}, engine, True, "low", 500)
    assert job.ids == enc(prompt)
    reply = enc(text[len(prompt):])[:-1]  # without the end-of-turn token
    decoder.scripts.append((list(job.ids), reply))
    thought, streamed, result = serve(engine, job)[0]
    assert streamed == result["content"] == "확인해 보겠습니다." and result["finish_reason"] == "tool_calls"
    assert thought == result["reasoning_content"] == said.get("reasoning_content", "")
    (got,) = result["tool_calls"]
    assert got["id"].startswith("call_") and got["function"]["name"] == "get_weather"
    assert json.loads(got["function"]["arguments"]) == {"city": "Seoul", "days": 2}
    # the next request carries the call (with its thinking, which upstream keeps before a tool's result) and the
    # result back: it goes on from the end of the reply
    messages += [{**said, "tool_calls": [got]}, {"role": "tool", "tool_call_id": got["id"], "content": "맑음, 3도"}]
    job = request_job({"messages": messages, "tools": tools, "temperature": 0, "thinking": switch}, engine, True, "low", 500)
    thinking = enc("Clear, three degrees.") + [tok.convert_tokens_to_ids("</think>")] if think else []
    decoder.scripts.append((list(job.ids), thinking + enc("맑고 3도입니다.")))
    _, streamed, result = serve(engine, job)[0]
    assert streamed == "맑고 3도입니다." and result["tool_calls"] == [] and result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == len(enc(prompt)) + len(reply)


def test_the_http_layer_answers_as_the_openai_api_does(tok, tmp_path):
    from scripts.serve import Handler, request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok, tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.app = argparse.Namespace(engine=engine, model="test-model", created=1, think=True, effort="low", max_tokens=500)
    threads = [threading.Thread(target=server.serve_forever, daemon=True), threading.Thread(target=engine.run, daemon=True)]
    for t in threads:
        t.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = {"model": "x", "messages": [{"role": "user", "content": [{"type": "text", "text": "1 더하기 1은?"}]}], "temperature": 0}
        think_end = tok.convert_tokens_to_ids("</think>")
        decoder.scripts.append((list(request_job(body, engine, True, "low", 500).ids), enc("One plus one.") + [think_end] + enc("2입니다.")))

        def post(payload):
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=30)

        whole = json.load(post(body))
        assert whole["object"] == "chat.completion" and whole["model"] == "test-model" and whole["id"].startswith("chatcmpl-")
        choice = whole["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "2입니다.", "reasoning_content": "One plus one."}
        assert choice["finish_reason"] == "stop" and whole["usage"]["total_tokens"] == whole["usage"]["prompt_tokens"] + whole["usage"]["completion_tokens"]

        lines = post({**body, "stream": True, "stream_options": {"include_usage": True}}).read().decode().split("\n\n")
        data = [line[6:] for line in lines if line.startswith("data: ")]
        assert data[-1] == "[DONE]"
        chunks = [json.loads(d) for d in data[:-1]]
        assert all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks)
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
        assert deltas[0] == {"role": "assistant", "content": ""}
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "One plus one."
        assert "".join(d.get("content", "") for d in deltas) == "2입니다."
        assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == whole["usage"]["completion_tokens"]

        assert json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"] == "test-model"
        with pytest.raises(urllib.error.HTTPError) as refused:
            post({"messages": []})
        assert refused.value.code == 400 and "messages" in json.load(refused.value)["error"]["message"]
    finally:
        engine.stopping = True
        server.shutdown()
