# The chat's token sequence for DeepSeek-V4-Flash-0731 against upstream's encoding module: what the decoder holds after
# a conversation is what encode_messages writes for it by default, with the earlier replies' thinking dropped.
# Run with SCLM_MODEL=v4 (needs the checkpoint's tokenizer).
import os

import pytest
import torch

from sclm import CKPT, MODEL

pytestmark = pytest.mark.skipif(MODEL != "v4" or not os.path.exists(os.path.join(CKPT, "tokenizer.json")), reason="set SCLM_MODEL=v4")


class Decoder:
    """Records what it is fed. From the end of a generation prompt (a thinking tag) on it answers with the scripted
    reply, one token a call, then the end of the turn."""

    def __init__(self, vocab: int, eos: int, tags: tuple):
        self.fed, self.reply, self.vocab, self.eos, self.tags, self.on = [], [], vocab, eos, tags, False
        self.skip = 0  # thinking tags to pass before the generation prompt's (the one that stands for dropped thinking)

    def snapshot(self):
        return len(self.fed)

    def restore(self, kept, back):
        assert kept == len(self.fed) - back
        del self.fed[kept:]

    def logits(self, ids, pos):
        assert pos == len(self.fed)
        self.fed += ids.flatten().tolist()
        if not self.on and self.fed[-1] in self.tags:
            self.on, self.skip = self.skip == 0, max(0, self.skip - 1)
        out = torch.full((1, self.vocab), -1e9)
        if self.on and not self.reply:
            self.on = False
        out[0, self.reply.pop(0) if self.on else self.eos] = 0.0
        return out


# ahead: how many tokens of the earlier reply are fed again before the next message comes (the chat does that while it
# reads the message; None: not started)
@pytest.mark.parametrize("ahead", [None, 0, 3, 1000])
@pytest.mark.parametrize("think,effort,system", [(True, "low", ""), (True, "high", "You answer briefly."), (False, "low", "")])
def test_the_chat_feeds_what_upstream_encodes(think, effort, system, ahead):
    from transformers import AutoTokenizer

    import sclm.upstream  # noqa: F401
    from encoding_dsv4 import encode_messages
    from scripts.chat import Session

    tok = AutoTokenizer.from_pretrained(CKPT)
    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    decoder = Decoder(len(tok), tok.eos_token_id, tuple(tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")))
    session = Session(tok, decoder, 4096)
    session.think, session.effort, session.system = think, effort, system
    turns = [("Rust로 숫자를 양자화하는 코드를 써 줘.", "Let me think about rounding.", "fn q(x: f64) -> i32 { x.round() as i32 }"),
             ("고마워, 테스트도 추가해 줘.", "A unit test with assert_eq.", "#[test] fn t() { assert_eq!(q(1.6), 2); }"),
             ("끝.", "Nothing more to do.", "네.")]
    messages = [{"role": "system", "content": system}] if system else []
    assert session.drop
    for n, (question, reasoning, answer) in enumerate(turns):
        decoder.reply = (enc(reasoning) + [session.think_end] if think else []) + enc(answer)
        decoder.skip = 1 if think and n else 0
        if ahead is not None:
            left = [ahead]
            session.settle(lambda: left.__setitem__(0, left[0] - 1) or left[0] < 0)
        session.ask(question, 500, 0.0, 1.0, lambda kind, text: None)
        messages += [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer, **({"reasoning_content": reasoning} if think else {})}]
    want = enc(encode_messages(messages, "thinking" if think else "chat", reasoning_effort=effort))
    assert not think or want.count(session.think_start) == 1  # only the last reply keeps its thinking
    assert want[0] == tok.bos_token_id and want.count(tok.bos_token_id) == 1
    assert decoder.fed == want[:-1]  # the last end-of-turn token is fed with the next turn
