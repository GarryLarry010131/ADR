"""Readouts built from a configuration dictionary:

    adr            model, variant: excess | dense | pi0 | argmax
    deployed       model
    halfcrop       model
    matching_head  model: blip_base | blip_large | blip2 | flava, head: itm | itc
    likelihood     model: llava15_7b | qwen3vl_8b | qwen36_27b
    chat           model: llava15_7b | qwen3vl_8b | qwen36_27b

Encoders and MLLMs are shared between readouts of one run that name the same model.
"""
from .base import ChatReadout, ScoringReadout  # noqa: F401

_SHARED = {}


def _encoder(model, device):
    key = ("encoder", model)
    if key not in _SHARED:
        from adr import Encoder
        _SHARED[key] = Encoder(model, device=device)
    return _SHARED[key]


def _mllm(model, device):
    key = ("mllm", model)
    if key not in _SHARED:
        from .mllm import _Loaded
        _SHARED[key] = _Loaded(model, device)
    return _SHARED[key]


def build(cfg, device=None):
    """Return (readout, tag) for one configuration entry."""
    name = cfg["readout"]
    model = cfg["model"]
    if name == "adr":
        from .dual_encoder import ADR
        variant = cfg.get("variant", "excess")
        return ADR(model, variant, encoder=_encoder(model, device)), "adr_%s_%s" % (variant, model)
    if name == "deployed":
        from .dual_encoder import Deployed
        return Deployed(model, encoder=_encoder(model, device)), "deployed_%s" % model
    if name == "halfcrop":
        from .halfcrop import HalfCrop
        return HalfCrop(model, encoder=_encoder(model, device)), "halfcrop_%s" % model
    if name == "matching_head":
        from .matching_heads import MatchingHead
        head = cfg.get("head", "itm")
        return MatchingHead(model, head, device), "%s_%s" % (head, model)
    if name == "likelihood":
        from .mllm import Likelihood
        return Likelihood(model, loaded=_mllm(model, device)), "likelihood_%s" % model
    if name == "chat":
        from .mllm import Chat
        return Chat(model, loaded=_mllm(model, device)), "chat_%s" % model
    raise ValueError("unknown readout %r" % name)
