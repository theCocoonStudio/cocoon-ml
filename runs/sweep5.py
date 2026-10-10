# Sweep 05 (the design after round one against sweep 04; written 2026-10-08, not yet run): two layers
# (a pairing is within reach), the scrambled-pairs control read beside every meaning curve, the plateau
# rule on a held-out set, the identification excess over the sampling floor beside the error, every
# point placed by its distance from the training table and the moved table's own entropy. The remap
# knob stays off unless Izzy turns it on. One cell per process, one JSON line out. Numbers to the log
# only (the rule).
# Usage: sweep5.py <name> <forms> <nonterminals> <resolution> <width> <length> <windows> <depth> <radii_grains: a,b,c> <remap_count> <table_seed> <model_seed> <cap> <layers> [lr=0.3] [patience=3] [minimum=300]
import json, random, sys, time
from fractions import Fraction

from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, drifted, filled, form_loss_by_rank, scramble_extensions, train_until_plateau
from cocoonml.probe import fit, r_squared
from cocoonml.schema import delta_magnitude, entropy_floor_form, entropy_floor_meaning, generate, identification_error, identification_excess, incidence_delta, n_eff, remap

name = sys.argv[1]
forms, nts, res, width, length, windows, depth = map(int, sys.argv[2:9])
radii = [int(x) for x in sys.argv[9].split(",")]
remap_count, tseed, mseed, cap, layers = map(int, sys.argv[10:15])
LR = float(sys.argv[15]) if len(sys.argv) > 15 else 0.3
PATIENCE = int(sys.argv[16]) if len(sys.argv) > 16 else 3
MINIMUM = int(sys.argv[17]) if len(sys.argv) > 17 else 300
SEP, BASE = forms + 1, forms + 2
LEAN = Fraction(3, 4)


def strings_of(tokens, n):
    segments, current = [], []
    for t in tokens:
        if t == SEP:
            segments.append(tuple(current))
            current = []
        elif t < BASE:
            current.append(t)
    return segments[:n]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def meaning_by_rank(model, made, whole):
    """Mean cross-entropy of the extension targets by separator rank over prepared contexts, at the
    ranks every context reaches (the harness's reader, on contexts given rather than drawn, so the
    scrambled control reads the very same windows)."""
    import math
    sums, counts = {}, {}
    for tokens, targets in made:
        logits, _ = model.forward(tokens)
        rank = 0
        for tok, lg, t in zip(tokens, logits, targets):
            if tok == SEP and t >= BASE:
                m = max(x.value for x in lg)
                total = sum(math.exp(x.value - m) for x in lg)
                sums[rank] = sums.get(rank, 0.0) + (-(lg[t].value - m) + math.log(total))
                counts[rank] = counts.get(rank, 0) + 1
                rank += 1
    return [sums[r] / counts[r] for r in sorted(sums) if r < whole]


def read(model, table, rng, against):
    ws = [filled(table, SEP, length, rng, BASE, pairs=True) for _ in range(windows)]
    whole = min(n for _, _, n in ws)
    strings = [strings_of(tokens, n) for tokens, _, n in ws]
    ident = [identification_error(table, s) for s in strings]
    excess = [identification_excess(table, s)[0] for s in strings]
    shown = [identification_error(against, s)[0] for s in strings]
    half = ws[: windows // 2]
    meaning_curve = meaning_by_rank(model, [(t, g) for t, g, _ in half], whole)
    scrambled = meaning_by_rank(model, [(scramble_extensions(t, rng, BASE), g) for t, g, _ in half], whole)
    form_curve = form_loss_by_rank(model, table, windows // 2, 0, SEP, rng, fill=True, extension_base=BASE, pairs=True)
    xs, ys = [], []
    for (tokens, _, n), wd in zip(ws, shown):
        if wd is None or n == 0:
            continue
        seps = [i for i, t in enumerate(tokens) if t == SEP]
        xs.append(model.probe(tokens)[seps[n - 1]])
        ys.append(wd)
    floors = {"form": entropy_floor_form(table), "meaning": entropy_floor_meaning(table)}
    return {
        "floors": floors,
        "n_eff": mean(n_eff(table, s, 4)[0] for s in strings),
        "foreignness": mean(n_eff(against, s, 4)[0] for s in strings),
        "identification_error": mean(e for e, _, _ in ident),
        "identification_excess": mean(excess),
        "coverage": mean(c for _, c, _ in ident),
        "window_drift_shown": mean(shown),
        "artifacts_per_window": mean(n for _, _, n in ws),
        "form_by_rank": form_curve,
        "form_excess_by_rank": [x - floors["form"] for x in form_curve],
        "meaning_by_rank": meaning_curve,
        "meaning_excess_by_rank": [x - floors["meaning"] for x in meaning_curve],
        "meaning_scrambled_by_rank": scrambled,
        "scramble_cost_by_rank": [a - b for a, b in zip(scrambled, meaning_curve)],
    }, xs, ys


t0 = time.time()
table = generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=tseed)
bound = Fraction(nts, 2 * res)
rng = random.Random(100 * tseed + mseed)
model = Attention(vocab=BASE + 3 * length, width=width, length=length, seed=mseed, separator=SEP, layers=layers)
held_out = contexts_meaning(table, 16, 0, SEP, length, BASE, random.Random(999 + tseed), fill=True, pairs=True)
evaluate = lambda: sum(model.loss(t, g).value for t, g in held_out) / len(held_out)
losses, evaluations = train_until_plateau(model, lambda: contexts_meaning(table, 4, 0, SEP, length, BASE, rng, fill=True, pairs=True), LR, window=50, tolerance=0.01, cap=cap, evaluate=evaluate, patience=PATIENCE, minimum=MINIMUM)
same, xs, ys = read(model, table, rng, table)
points = {}
for r in radii:
    drift_rng = random.Random(1000 * tseed + 7 * r + 1)
    moved = drifted(table, depth, bound, LEAN, drift_rng, radius=Fraction(r, res) if r > 0 else None)
    moves = []
    if remap_count > 0:
        moved, moves = remap(moved, random.Random(5000 * tseed + r), remap_count)
    reading, x2, y2 = read(model, moved, rng, table)
    xs += x2
    ys += y2
    reading["coordinates"] = {"distance": delta_magnitude(table, moved), "incidence": float(incidence_delta(table, moved)), "entropy_form": reading["floors"]["form"], "entropy_meaning": reading["floors"]["meaning"]}
    reading["remaps"] = moves
    reading["form_recovery_vs_same"] = [a - b for a, b in zip(reading["form_excess_by_rank"], same["form_excess_by_rank"])]
    reading["meaning_recovery_vs_same"] = [a - b for a, b in zip(reading["meaning_excess_by_rank"], same["meaning_excess_by_rank"])]
    points[r] = reading
r2 = None
idx = list(range(len(xs)))
random.Random(0).shuffle(idx)
half = len(idx) // 2
if half >= width + 2:
    w, b = fit([xs[i] for i in idx[:half]], [ys[i] for i in idx[:half]], ridge=1e-2)
    r2 = r_squared(w, b, [xs[i] for i in idx[half:]], [ys[i] for i in idx[half:]])
print(json.dumps({
    "run": name, "args": sys.argv[1:], "table": tseed, "model": mseed, "layers": layers, "seconds": round(time.time() - t0, 1),
    "train_steps": len(losses), "hit_cap": len(losses) >= cap, "evaluations": evaluations, "train_first": losses[0], "train_last": losses[-1],
    "same": same, "points": points, "probe_r2_heldout": r2, "probe_windows": len(xs),
}))
