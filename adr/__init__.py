"""Antisymmetric Displacement Readout (ADR). The encoders need torch and open_clip and are imported lazily."""
from .lexicon import DEPTH, LEX, LEX_VSR, LR_RELS, grid, parse
from .readout import (VARIANTS, all_scores, column_weights, derangement, derangement_grouped, displacement, fair,
                      kway_fair, oracle_accuracy, score, sign_fair)
from .sinkhorn import EPS, TOL, sinkhorn_max

try:
    from .features import MODELS, Encoder
except ImportError:
    MODELS, Encoder = None, None

__all__ = ["MODELS", "Encoder", "DEPTH", "LEX", "LEX_VSR", "LR_RELS", "grid", "parse", "VARIANTS", "all_scores",
           "column_weights", "derangement", "derangement_grouped", "displacement", "fair", "kway_fair",
           "oracle_accuracy", "score", "sign_fair", "EPS", "TOL", "sinkhorn_max"]
