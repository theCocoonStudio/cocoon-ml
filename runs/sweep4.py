# Sweep 04 (2026-10-08), the design after round one against sweeps 03: pairs in the input (meaning has an
# in-context channel), a within-artifact position, training to a plateau, the resolution lowered and
# the context lengthened past the count's identification threshold, the RADIUS swept at a fixed depth
# past mixing with the measured distance as the x-axis, the remap knob on the incidence (0 = off, the
# design as Izzy set it; on only at their word), the same-table curves subtracted in the script.
# One cell per process, one JSON line out. Numbers go to the log only (the rule).
# Usage: sweep4.py <name> <forms> <nonterminals> <resolution> <width> <length> <windows> <depth> <radii_grains: a,b,c> <remap_count> <table_seed> <model_seed> <cap>
import json, random, sys, time
from fractions import Fraction

from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, drifted, filled, form_loss_by_rank, meaning_loss_at_separators, train_until_plateau
from cocoonml.probe import fit, r_squared
from cocoonml.schema import delta_magnitude, entropy_floor_form, entropy_floor_meaning, generate, identification_error, incidence_delta, n_eff, remap

name = sys.argv[1]
forms, nts, res, width, length, windows, depth = map(int, sys.argv[2:9])
radii = [int(x) for x in sys.argv[9].split(",")]
remap_count, tseed, mseed, cap = map(int, sys.argv[10:14])
SEP, BASE = forms + 1, forms + 2
LEAN = Fraction(3, 4)


def strings_of(tokens, n):
    """The first n artifacts of a packed context with pairs, as strings (extension tokens dropped)."""
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


def read(model, table, rng, against):
    """Everything read on one table: the windows' readers, the curves, the probe rows."""
    ws = [filled(table, SEP, length, rng, BASE, pairs=True) for _ in range(windows)]
    strings = [strings_of(tokens, n) for tokens, _, n in ws]
    ident = [identification_error(table, s) for s in strings]
    shown = [identification_error(against, s)[0] for s in strings]
    form_curve = form_loss_by_rank(model, table, windows // 2, 0, SEP, rng, fill=True, extension_base=BASE, pairs=True)
    meaning_curve = meaning_loss_at_separators(model, table, windows // 2, 0, SEP, length, BASE, rng, fill=True, pairs=True)
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
        "coverage": mean(c for _, c, _ in ident),
        "window_drift_shown": mean(shown),
        "artifacts_per_window": mean(n for _, _, n in ws),
        "form_by_rank": form_curve,
        "form_excess_by_rank": [x - floors["form"] for x in form_curve],
        "meaning_by_rank": meaning_curve,
        "meaning_excess_by_rank": [x - floors["meaning"] for x in meaning_curve],
    }, xs, ys


t0 = time.time()
table = generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=tseed)
bound = Fraction(nts, 2 * res)
rng = random.Random(100 * tseed + mseed)
model = Attention(vocab=BASE + 3 * length, width=width, length=length, seed=mseed, separator=SEP)
losses = train_until_plateau(model, lambda: contexts_meaning(table, 4, 0, SEP, length, BASE, rng, fill=True, pairs=True), 0.3, window=50, tolerance=0.01, cap=cap)
same, xs, ys = read(model, table, rng, table)
per_radius = {}
for r in radii:
    drift_rng = random.Random(1000 * tseed + 7 * r + 1)
    moved = drifted(table, depth, bound, LEAN, drift_rng, radius=Fraction(r, res) if r > 0 else None)
    moves = []
    if remap_count > 0:
        moved, moves = remap(moved, random.Random(5000 * tseed + r), remap_count)
    reading, x2, y2 = read(model, moved, rng, table)
    xs += x2
    ys += y2
    reading["delta_magnitude"] = delta_magnitude(table, moved)
    reading["incidence_delta"] = float(incidence_delta(table, moved))
    reading["remaps"] = moves
    reading["form_recovery_vs_same"] = [a - b for a, b in zip(reading["form_excess_by_rank"], same["form_excess_by_rank"])]
    reading["meaning_recovery_vs_same"] = [a - b for a, b in zip(reading["meaning_excess_by_rank"], same["meaning_excess_by_rank"])]
    per_radius[r] = reading
r2 = None
idx = list(range(len(xs)))
random.Random(0).shuffle(idx)
half = len(idx) // 2
if half >= width + 2:
    w, b = fit([xs[i] for i in idx[:half]], [ys[i] for i in idx[:half]], ridge=1e-2)
    r2 = r_squared(w, b, [xs[i] for i in idx[half:]], [ys[i] for i in idx[half:]])
print(json.dumps({
    "run": name, "args": sys.argv[1:], "table": tseed, "model": mseed, "seconds": round(time.time() - t0, 1),
    "train_steps": len(losses), "hit_cap": len(losses) >= cap, "train_first": losses[0], "train_mid": losses[len(losses) // 2], "train_last": losses[-1],
    "same": same, "per_radius": per_radius, "probe_r2_heldout": r2, "probe_windows": len(xs),
}))
