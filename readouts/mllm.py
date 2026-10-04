"""MLLM readouts. Likelihood: log-probability of each candidate caption after a fixed instruction, summed over the
caption tokens (oracle threshold on true/false sets). Chat: one-letter or yes/no answers, greedy, four new tokens,
asked in both candidate orders and averaged; an unparsable reply earns the uniform credit."""
import re

import torch
import torch.nn.functional as F

from .base import ChatReadout, ScoringReadout

MODEL_ID = {"llava15_7b": "llava-hf/llava-1.5-7b-hf", "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct", "qwen36_27b": "Qwen/Qwen3.6-27B"}
PREFIX = "Write a one-sentence description of the image."
Q_KWAY = "Which of the following captions correctly describes the image?\n{options}\nAnswer with a single letter, {choices}."
Q_TRUEFALSE = "Is the following statement true of the image?\n{statement}\nAnswer with a single word, yes or no."
LETTERS = "ABCDEFGH"
YES_RE = re.compile(r"\b(YES|NO)\b")


class _Loaded:
    def __init__(self, model, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        assert model in MODEL_ID, "unknown MLLM %r, choose from %s" % (model, sorted(MODEL_ID))
        self.key, self.hf_id = model, MODEL_ID[model]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28} if model.startswith("qwen3vl") else {}
        self.proc = AutoProcessor.from_pretrained(self.hf_id, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=torch.bfloat16, device_map=self.device).eval()
        self.template_kw = {"enable_thinking": False} if model.startswith("qwen36") else {}

    def prompt(self, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self.template_kw)

    @torch.no_grad()
    def generate(self, img, text):
        enc = self.proc(text=self.prompt(text), images=[img], return_tensors="pt").to(self.device)
        pad = getattr(self.proc.tokenizer, "pad_token_id", None) or self.proc.tokenizer.eos_token_id
        out = self.model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=pad)
        return self.proc.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()


class Likelihood(ScoringReadout):
    kind = "score"

    def __init__(self, model="qwen36_27b", device=None, loaded=None):
        self.m = loaded or _Loaded(model, device)

    @torch.no_grad()
    def scores(self, img, captions):
        pre = self.m.prompt(PREFIX)
        L0 = self.m.proc(text=pre, images=[img], return_tensors="pt").input_ids.shape[1]
        out = []
        for cap in captions:
            enc = self.m.proc(text=pre + cap, images=[img], return_tensors="pt").to(self.m.device)
            L = enc.input_ids.shape[1]
            assert L > L0, "caption produced no tokens"
            logits = self.m.model(**enc).logits[0]
            lp = F.log_softmax(logits[L0 - 1:L - 1].float(), dim=-1)
            tgt = enc.input_ids[0, L0:L]
            out.append(float(lp.gather(-1, tgt[:, None]).sum()))
        return out


class Chat(ChatReadout):
    def __init__(self, model="qwen36_27b", device=None, loaded=None):
        self.m = loaded or _Loaded(model, device)

    def choose(self, img, captions, correct):
        credits, log = [], []
        for rev in (False, True):
            idx = list(range(len(captions)))[::-1] if rev else list(range(len(captions)))
            shown = [captions[i] for i in idx]
            letters = LETTERS[:len(shown)]
            choices = (", ".join(letters[:-1]) + ", or " + letters[-1]) if len(letters) > 2 else " or ".join(letters)
            q = Q_KWAY.format(options="\n".join("%s: %s" % (letters[i], shown[i]) for i in range(len(shown))), choices=choices)
            reply = self.m.generate(img, q)
            m = re.search(r"\b([%s])\b" % letters, reply)
            if m is None:
                credits.append(1.0 / len(captions))
                log.append({"reply": reply, "parsed": False, "chose": None})
            else:
                chosen = idx[letters.index(m.group(1))]
                credits.append(1.0 if chosen == correct else 0.0)
                log.append({"reply": reply, "parsed": True, "chose": chosen})
        return sum(credits) / len(credits), log

    def yesno(self, img, statement, label):
        reply = self.m.generate(img, Q_TRUEFALSE.format(statement=statement))
        m = YES_RE.search(reply)
        if m is None:
            return 0.5, {"reply": reply, "parsed": False, "said": None}
        said = 1 if m.group(1) == "YES" else 0
        return (1.0 if said == label else 0.0), {"reply": reply, "parsed": True, "said": said}
