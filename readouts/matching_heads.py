"""Cross-encoder matching heads (BLIP, BLIP-2, FLAVA): the log-softmax matched-class probability of the ITM head, or
the contrastive similarity of the same checkpoint. Plain scores, decided by an oracle threshold on true/false sets."""
import numpy as np
import torch

from .base import ScoringReadout

REGISTRY = {"blip_base": ("Salesforce/blip-itm-base-coco", "blip"), "blip_large": ("Salesforce/blip-itm-large-coco", "blip"),
            "blip2": ("Salesforce/blip2-itm-vit-g-coco", "blip2"), "flava": ("facebook/flava-full", "flava")}
MATCH_COL = 1


class MatchingHead(ScoringReadout):
    kind = "score"

    def __init__(self, model="blip_base", head="itm", device=None):
        assert model in REGISTRY and head in ("itm", "itc")
        self.hf_id, self.family = REGISTRY[model]
        self.head = head
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.family == "blip":
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            self.proc = BlipProcessor.from_pretrained(self.hf_id)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.hf_id, dtype=torch.float32)
        elif self.family == "blip2":
            from transformers import Blip2ForImageTextRetrieval, Blip2Processor
            self.proc = Blip2Processor.from_pretrained(self.hf_id)
            self.model = Blip2ForImageTextRetrieval.from_pretrained(self.hf_id, dtype=torch.float32)
        else:
            from transformers import FlavaForPreTraining, FlavaProcessor
            self.proc = FlavaProcessor.from_pretrained(self.hf_id)
            self.model = FlavaForPreTraining.from_pretrained(self.hf_id, dtype=torch.float32)
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def _both(self, images, captions):
        n = len(images)
        if self.family == "blip":
            enc = self.proc(images=images, text=captions, return_tensors="pt", padding=True).to(self.device)
            itm_logits = self.model(**enc, use_itm_head=True).itm_score
            sim = self.model(**enc, use_itm_head=False).itm_score
        elif self.family == "blip2":
            enc = self.proc(images=images, text=captions, return_tensors="pt", padding=True).to(self.device)
            itm_logits = self.model(**enc, use_image_text_matching_head=True).logits_per_image
            sim = self.model(**enc, use_image_text_matching_head=False).logits_per_image
        else:
            enc = self.proc(images=images, text=captions, return_tensors="pt", padding=True,
                            return_codebook_pixels=False, return_image_mask=False).to(self.device)
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], pixel_values=enc["pixel_values"],
                             input_ids_masked=enc["input_ids"], skip_unmasked_multimodal_encoder=True, return_loss=False)
            itm_logits, sim = out.itm_logits, out.contrastive_logits_per_image
        assert tuple(itm_logits.shape) == (n, 2) and tuple(sim.shape) == (n, n)
        itm = torch.log_softmax(itm_logits.float(), dim=1)[:, MATCH_COL]
        itc = torch.diagonal(sim.float())
        return itm.cpu().numpy().astype(np.float64), itc.cpu().numpy().astype(np.float64)

    def scores(self, img, captions):
        itm, itc = self._both([img] * len(captions), list(captions))
        return list(itm if self.head == "itm" else itc)
