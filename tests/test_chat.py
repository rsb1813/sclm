# Checks the chat's conversation logic (scripts/chat.py) with a stand-in tokenizer and decoder: what is fed at which
# position, how the reply is split and streamed, and the token choice.
import pytest
import torch

from types import SimpleNamespace

from rich.console import Console

from scripts.chat import ContextFull, Session, command, looping, pick

SPECIAL = {"<bos>": 0, "<eos>": 1, "<think>": 2, "</think>": 3, "<user>": 4, "<assistant>": 5, "<system>": 6}
NAMES = {i: s for s, i in SPECIAL.items()}
HALF_A, HALF_B = 50, 51  # two tokens that make one character only together
VOCAB = 300


class Tok:
    eos_token, eos_token_id = "<eos>", 1

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, thinking_mode, reasoning_effort):
        assert add_generation_prompt and not tokenize
        text = "<bos>" + f"[{reasoning_effort}]"
        for msg in messages:
            text += f"<{msg['role']}>{msg['content']}" + ("<eos>" if msg["role"] == "assistant" else "")
        return text + "<assistant>" + ("<think>" if thinking_mode == "thinking" else "</think>")

    def encode(self, text, add_special_tokens=True):
        ids = []
        while text:
            for s, i in SPECIAL.items():
                if text.startswith(s):
                    ids.append(i)
                    text = text[len(s):]
                    break
            else:
                ids.append(100 + ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids, clean_up_tokenization_spaces=False):
        out = ""
        for i in ids:
            if i == HALF_B and out.endswith("�"):
                out = out[:-1] + "é"
            else:
                out += "�" if i in (HALF_A, HALF_B) else NAMES[i] if i in NAMES else chr(i - 100)
        return out


class Decoder:
    """Records what it is fed. From the end of a generation prompt (a thinking tag) to the next end-of-turn token it
    answers with logits that make the scripted tokens the most likely ones in turn."""

    def __init__(self, script):
        self.fed, self.script, self.said, self.on = [], list(script), 0, False

    def logits(self, ids, pos):
        token = int(ids)
        self.fed.append((token, pos))
        self.on = token != 1 and (self.on or token in (SPECIAL["<think>"], SPECIAL["</think>"]))
        out = torch.zeros(1, VOCAB)
        if self.on:
            out[0, self.script[self.said] if self.said < len(self.script) else 1] = 9.0
            self.said += 1
        return out


def text_ids(text):
    return Tok().encode(text)


def run(session, message, **kw):
    pieces = []
    stats = session.ask(message, kw.pop("max_new", 100), 0.0, 1.0, lambda kind, text: pieces.append((kind, text)), **kw)
    return pieces, stats


def test_a_turn_is_fed_once_and_the_reply_is_split_at_the_end_of_thinking():
    script = text_ids("hm") + [3] + text_ids("ok") + [1]
    dec = Decoder(script)
    session = Session(Tok(), dec, 200)
    session.system = "S"
    pieces, stats = run(session, "hi")
    prompt = text_ids("<bos>[high]<system>S<user>hi<assistant><think>")
    assert [t for t, _ in dec.fed] == prompt + script[:-1]  # the end-of-turn token is not fed
    assert [p for _, p in dec.fed] == list(range(len(dec.fed)))
    assert "".join(t for k, t in pieces if k == "think") == "hm" and "".join(t for k, t in pieces if k == "answer") == "ok"
    assert ("answer", "") in pieces  # the change is announced, the tag itself is not shown
    assert stats["tokens"] == len(script) - 1 and stats["prompt_tokens"] == len(prompt)

    dec.script, dec.said = text_ids("yo") + [1], 0  # the next turn goes on in the same sequence
    session.think = False
    before = len(dec.fed)
    pieces, _ = run(session, "more")
    assert [t for t, _ in dec.fed[before:]] == [1] + text_ids("<user>more<assistant></think>") + text_ids("yo")
    assert dec.fed[before][1] == before and pieces == [("answer", "y"), ("answer", "o")]

    session.clear()
    dec.script, dec.said = [1], 0
    run(session, "new")
    assert dec.fed[-1][1] == len(text_ids("<bos>[high]<system>S<user>new<assistant></think>")) - 1 and dec.fed[-1][0] == 3


def test_a_character_is_shown_when_all_its_tokens_are_there():
    dec = Decoder([3, 100 + ord("a"), HALF_A, HALF_B, 1])
    pieces, _ = run(Session(Tok(), dec, 200), "x")
    assert [t for k, t in pieces if t] == ["a", "é"]


def test_stopping_and_limits_end_the_reply_and_the_context_can_fill_up():
    dec = Decoder([3] + text_ids("abcdef") + [1])
    session = Session(Tok(), dec, 200)
    seen = []
    pieces, stats = run(session, "x", stop=lambda: len(seen) >= 2 or seen.append(1))
    assert stats["tokens"] == 2  # asked before every token: two went through
    dec.said = 0
    pieces, stats = run(session, "y", max_new=3)
    assert stats["tokens"] == 3 and dec.fed[-4 - len(text_ids("<user>y<assistant><think>"))][0] == 1

    small = Session(Tok(), Decoder([1]), 20)
    with pytest.raises(ContextFull):
        run(small, "a message that does not fit")
    assert small.pos == 0


def test_pick_takes_the_likeliest_token_or_samples_within_top_p():
    logits = torch.tensor([[0.0, 3.0, 2.9, -1.0]])
    assert pick(logits, 0.0, 1.0) == 1
    assert {pick(logits, 1.0, 0.1) for _ in range(20)} == {1}  # top_p below the first probability keeps only it
    assert {pick(logits, 1.0, 0.9) for _ in range(200)} == {1, 2}


def test_a_loop_is_recognized_and_ends_the_reply():
    piece = text_ids("Rust has x? ")
    assert looping(text_ids("so: ") + piece * 8) and looping([7] * 96) and looping([7, 8] * 48)
    assert not looping(text_ids("so: ") + piece * 7)  # 84 tokens of it: not yet
    assert not looping(list(range(40)) * 5)  # a piece longer than 32 tokens is not taken for a loop
    assert not looping([7] * 95 + [8]) and not looping([])

    dec = Decoder([3] + piece * 50 + [1])
    session = Session(Tok(), dec, 2000)
    pieces, stats = run(session, "x", max_new=1000)
    assert stats["looped"] and stats["tokens"] == 1 + 8 * len(piece)  # the end of thinking, then eight times the piece
    dec.script, dec.said = text_ids("ok") + [1], 0
    pieces, stats = run(session, "y")
    assert not stats["looped"] and "".join(t for _, t in pieces) == "ok"


def test_a_setting_is_a_command_of_its_own():
    session, cache = Session(Tok(), Decoder([1]), 200), SimpleNamespace(wait_share=0.0, topk=0)
    opts = {"temperature": 1.0, "top_p": 0.95, "num_predict": 2048, "verbose": True, "showthink": False}
    console = Console(force_terminal=False, width=200, record=True)
    for line in ("/nothink", "/effort low", "/temperature 0.5", "/topk 5", "/wait-share 0.25", "/showthink", "/quiet",
                 "/system be brief", "/num_predict 64", "/show"):
        assert command(line, session, cache, opts, console)
    assert (session.think, session.effort, session.system) == (False, "low", "be brief")
    assert (opts["temperature"], opts["num_predict"], opts["showthink"], opts["verbose"]) == (0.5, 64, True, False)
    assert (cache.topk, cache.wait_share) == (5, 0.25)
    shown = console.export_text()  # what /show printed
    assert "temperature 0.5" in shown and "topk 5" in shown
    for line in ("/set think", "/set topk 4", "/set"):  # the older form means the same
        assert command(line, session, cache, opts, console)
    assert session.think and cache.topk == 4
    for line in ("/effort huge", "/temperature hot", "/nonsense"):  # refused, and nothing changes
        assert command(line, session, cache, opts, console)
    assert session.effort == "low" and opts["temperature"] == 0.5
    assert not command("/bye", session, cache, opts, console)
