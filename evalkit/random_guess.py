"""The random-guess row: 10,000 uniform draws per set, scored under the published rule. The paper's row came from
one random stream (seed 0) over the sets in PAPER_ORDER, which `paper_row()` reproduces.

    python -m evalkit.random_guess
"""
import numpy as np

R, SEED = 10000, 0
PAPER_ORDER = [("vgrel_leftright", 3918, 2), ("vgrel_vertical", 800, 2), ("coco_spatial", 404, 2), ("whatsup_a", 408, 4),
               ("whatsup_b_front_behind", 176, 2), ("vsr_horizontal", 1201, 2), ("vgrel_verbs", 408, 2), ("valse_actant_swap", 949, 2),
               ("whatsup_b_left_right", 178, 2), ("whatsup_b", 354, 2)]


def draw(rng, n, k, R=R):
    correct = rng.integers(0, k, size=n)
    guess = rng.integers(0, k, size=(R, n))
    acc = (guess == correct[None, :]).mean(1)
    return {"n": n, "k": k, "mean": float(acc.mean()), "sd_single_draw": float(acc.std(ddof=1)),
            "p2_5": float(np.percentile(acc, 2.5)), "p97_5": float(np.percentile(acc, 97.5))}


def paper_row(seed=SEED):
    rng = np.random.default_rng(seed)
    return {name: draw(rng, n, k) for name, n, k in PAPER_ORDER}


def random_guess(n, k, seed=SEED):
    return draw(np.random.default_rng(seed), n, k)


if __name__ == "__main__":
    for name, r in paper_row().items():
        print("%-24s n=%4d k=%d  mean %.4f  one draw 95%% [%.4f, %.4f]" % (name, r["n"], r["k"], r["mean"], r["p2_5"], r["p97_5"]))
