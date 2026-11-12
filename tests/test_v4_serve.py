# The server's engine and HTTP layer (scripts/serve.py) on a scripted decoder: what a conversation's sequence holds
# against upstream's encoding, requests taking turns, the room they share, stop strings, tool calls and the responses.
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

    def reserve(self, tokens):
        self.reserved.append(tokens)

    def sequence(self, capacity):
        return Seq(capacity)

    def switch(self, seq):
        self.seq = seq

    def resize(self, seq, capacity):
        seq.capacity = capacity

    def clone(self, seq, capacity):
        new = Seq(capacity)
        new.fed = list(seq.fed)
        return new

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


def made(tok, context=1 << 16, **kw):
    from scripts.serve import Engine

    decoder = Decoder(len(tok), tok.eos_token_id)
    return Engine(tok, decoder, context, **kw), decoder


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
def test_a_conversation_keeps_its_sequence_and_holds_what_upstream_encodes(tok, think):
    import sclm.upstream  # noqa: F401
    from encoding_dsv4 import encode_messages
    from scripts.serve import request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok)
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
    assert engine.idle[0].seq.fed == want[:-1] and len(set(map(id, decoder.order))) == 1


def test_requests_open_together_take_turns_and_share_the_room(tok):
    from scripts.serve import CHUNK, Job

    engine, decoder = made(tok, context=2 * CHUNK, slice=2)
    jobs = []
    for n in range(3):  # each needs a chunk of room: two are open together, the third waits for one of them to end
        prompt, reply = [10 + n] * (7 + n), [20 + n] * (6 + 2 * n)
        decoder.scripts.append((prompt, reply))
        jobs.append(Job(prompt, 100, temperature=0.0))
    results = serve(engine, *jobs)
    for n, (job, (_, _, result)) in enumerate(zip(jobs, results)):
        assert job.conv.seq.fed == [10 + n] * (7 + n) + [20 + n] * (6 + 2 * n) and result["finish_reason"] == "stop"
    first, second, third = (job.conv.seq for job in jobs)
    order = decoder.order
    # a prompt goes in together up to its last token, whatever the slice; from there the turns are two tokens long
    assert order[:21] == [first] * 6 + [second] * 7 + [first, first, second, second] * 2
    assert third not in order[:order.index(third)] and first not in order[order.index(third):]  # it began when the first ended
    assert engine.room() == 0 and len(engine.idle) == 2  # the first one's sequence made room for the third


def test_a_reply_makes_room_as_it_grows_up_to_the_context(tok):
    import random

    from scripts.serve import CHUNK, Job

    engine, decoder = made(tok, context=4 * CHUNK, keep=0, reserve=CHUNK)
    words = random.Random(0).choices(range(1000, 2000), k=5 * CHUNK)  # no loop in them
    decoder.scripts += [([5, 6, 7], words[:2 * CHUNK + 9]), ([8, 9], words), ([3], words[:4])]
    grown, full, small = (Job(prompt, 10 * CHUNK, temperature=0.0) for prompt, _ in decoder.scripts)
    (_, _, result), = serve(engine, grown)
    assert result["finish_reason"] == "stop" and result["usage"]["completion_tokens"] == 2 * CHUNK + 9
    assert decoder.reserved == [CHUNK, 2 * CHUNK, 4 * CHUNK]  # doubled as the sequence passed a chunk, then two
    (_, _, result), = serve(engine, full)
    assert result["finish_reason"] == "length" and result["usage"]["total_tokens"] == 4 * CHUNK == len(full.conv.seq.fed)
    serve(engine, small)
    # nothing is kept: each request starts from the least again
    assert decoder.reserved == [CHUNK, 2 * CHUNK, 4 * CHUNK] * 2 + [CHUNK] and engine.room() == 4 * CHUNK


def test_a_long_prompt_goes_in_as_runs_up_to_its_last_token(tok):
    from scripts.serve import RUN, Job

    engine, decoder = made(tok, context=4 * RUN)
    long, short = [11] * (RUN + 700) + [12], [13] * 300 + [14]
    decoder.scripts += [(long, [21, 22]), (short, [23])]
    (_, _, first), (_, _, second) = serve(engine, Job(long, 10, temperature=0.0), Job(short, 10, temperature=0.0))
    assert decoder.runs == [RUN, 700] and first["usage"]["completion_tokens"] == 2 and second["usage"]["completion_tokens"] == 1
    assert engine.idle[0].seq.fed == long + [21, 22] and sorted(engine.idle[0].marks) == [RUN, RUN + 700]


def test_another_question_on_the_same_document_goes_on_from_the_documents_last_run(tok):
    from scripts.serve import RUN, Job

    engine, decoder = made(tok, context=4 * RUN)
    document = [11] * (2 * RUN + 700)
    first, second = document + [12] * 5 + [13], document + [14] * 9 + [15]
    decoder.scripts += [(first, [21, 22]), (second, [23])]
    (_, _, one), = serve(engine, Job(first, 10, temperature=0.0))
    assert decoder.runs == [RUN, RUN, 705] and one["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    (_, _, two), = serve(engine, Job(second, 10, temperature=0.0))
    # the two prompts part inside the third run: the sequence goes back to the end of the second
    assert two["usage"]["prompt_tokens_details"]["cached_tokens"] == 2 * RUN and decoder.runs[3:] == [709]
    assert len(engine.idle) == 1 and engine.idle[0].seq.fed == second + [23]
    assert sorted(engine.idle[0].marks) == [RUN, 2 * RUN, 2 * RUN + 709]


def test_sessions_with_the_same_system_prompt_each_go_on_from_a_copy_of_it(tok):
    from scripts.serve import RUN, TAIL, Job

    engine, decoder = made(tok, context=8 * RUN)
    system = [31] * 900
    one, two, three = (system + [40 + k] * n + [50 + k] for k, n in enumerate((TAIL + 500, 60, 700)))
    decoder.scripts += [(one, [61]), (two, [62]), (three, [63])]
    serve(engine, Job(one, 10, temperature=0.0, marks=[900], pins=[900]))
    first = engine.idle[0]
    # the second session parts from the first after the system prompt, where the first has a long way behind it
    (_, _, result), = serve(engine, Job(two, 10, temperature=0.0, marks=[900], pins=[900]))
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert len(engine.idle) == 2 and first.seq.fed == one + [61] and engine.idle[1].seq.fed == two + [62]
    # two requests that come together: the second waits for the first to feed what they share, then copies it
    again = system + [47] * (RUN + 600)
    a, b = again + [48] * 30 + [58], again + [49] * 30 + [59]
    decoder.scripts += [(a, [64]), (b, [65])]
    before = len(decoder.runs)
    (_, _, ra), (_, _, rb) = serve(engine, Job(a, 10, temperature=0.0, marks=[900], pins=[900]),
                                   Job(b, 10, temperature=0.0, marks=[900], pins=[900]))
    assert ra["usage"]["prompt_tokens_details"]["cached_tokens"] == 900
    assert rb["usage"]["prompt_tokens_details"]["cached_tokens"] == 900 + RUN  # the end of the first one's first run
    assert decoder.runs[before:] == [RUN, 630, 630] and first.seq.fed == one + [61]


def test_a_request_can_ask_for_a_mark_after_a_part_of_a_message(tok):
    from scripts.serve import request_job

    engine, _ = made(tok)
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


def test_stop_strings_and_the_token_limit_end_a_reply(tok):
    from scripts.serve import Job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok)
    reply = enc("alpha beta gamma delta epsilon")
    decoder.scripts += [([5, 6], reply), ([7, 8], reply)]
    stopped, cut = serve(engine, Job([5, 6], 100, temperature=0.0, stop=("gam", " delta eps")),
                         Job([7, 8], 3, temperature=0.0, stop=("zeta",)))
    assert stopped[1] == stopped[2]["content"] == "alpha beta " and stopped[2]["finish_reason"] == "stop"
    assert cut[2]["finish_reason"] == "length" and cut[2]["usage"]["completion_tokens"] == 3
    assert cut[1] == cut[2]["content"] == tok.decode(reply[:3])


def test_a_tool_call_is_parsed_out_of_the_answer(tok):
    import sclm.upstream  # noqa: F401
    from encoding_dsv4 import encode_messages
    from scripts.serve import request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
    messages = [{"role": "user", "content": "서울 날씨 알려 줘."}]
    call = {"type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Seoul", "days": 2})}}
    said = {"role": "assistant", "content": "확인해 보겠습니다.", "tool_calls": [call]}
    # upstream's own text for the assistant's turn: what follows the prompt in the encoding of the conversation with it
    system = {"role": "system", "content": "", "tools": tools}
    prompt = encode_messages([system] + messages, "chat")
    text = encode_messages([system] + messages + [said], "chat")
    assert text.startswith(prompt)
    job = request_job({"messages": messages, "tools": tools, "temperature": 0, "thinking": {"type": "disabled"}}, engine, True, "low", 500)
    assert job.ids == enc(prompt)
    decoder.scripts.append((list(job.ids), enc(text[len(prompt):])[:-1]))  # without the end-of-turn token
    _, streamed, result = serve(engine, job)[0]
    assert streamed == result["content"] == "확인해 보겠습니다." and result["finish_reason"] == "tool_calls"
    (got,) = result["tool_calls"]
    assert got["id"].startswith("call_") and got["function"]["name"] == "get_weather"
    assert json.loads(got["function"]["arguments"]) == {"city": "Seoul", "days": 2}
    # the next request carries the call and its result back: the sequence goes on from what it holds
    messages += [{**said, "tool_calls": [got]}, {"role": "tool", "tool_call_id": got["id"], "content": "맑음, 3도"}]
    job = request_job({"messages": messages, "tools": tools, "temperature": 0, "thinking": {"type": "disabled"}}, engine, True, "low", 500)
    decoder.scripts.append((list(job.ids), enc("맑고 3도입니다.")))
    _, streamed, result = serve(engine, job)[0]
    assert streamed == "맑고 3도입니다." and result["tool_calls"] == [] and result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > len(enc(prompt))


def test_the_http_layer_answers_as_the_openai_api_does(tok):
    from scripts.serve import Handler, request_job

    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    engine, decoder = made(tok)
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
