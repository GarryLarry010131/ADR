"""Decision rules and the derangement null shared by every readout."""
from adr.readout import TOL, derangement, derangement_grouped, fair, kway_fair, oracle_accuracy, sign_fair  # noqa: F401

# sets whose published null was drawn within groups of items
GROUP_KEY = {"whatsup_b": "task"}


def null_assignment(name, items, K):
    paths = [it["image"] for it in items]
    g = GROUP_KEY.get(name)
    if g:
        return derangement_grouped(paths, [it[g] for it in items], K)
    return derangement(paths, K)


__all__ = ["TOL", "derangement", "derangement_grouped", "fair", "kway_fair", "oracle_accuracy", "sign_fair", "GROUP_KEY", "null_assignment"]
