# Sweep 07 (2026-10-09, the complete system's first script; run at the derived size on the array
# model since 2026-10-09 afternoon): the corpus layer in place of the harness's contexts. Several
# graded tables in play, each walking within the training radius; a finite corpus produced in order
# with drawn steps and switches; windows as stretches of production (states succeed, tables mix, no
# extension token); training samples windows in no order and ends on the held-out plateau rule;
# readings on corpora walked to each read radius (grains; "free" for an unbounded walk). Readings:
# the form curve by rank and the order control, pooled per reading corpus and binned by the
# windows' distance (the reader's, from forms alone, and the truth's, with and without phases: a
# corpus walked at a radius holds states from the origin outward, so the curve past the training
# radius is read on the far bins, not the mixture; Methuselah's audit, 2026-10-09); the reader's
# distance per window against the truth's; two probes from the separator vectors, to the truth's
# distance and to the rank k, fitted on half the windows and read on the other half. The lean is an
# argument (swept, not a constant). The model is the array model at the derived initialisation,
# every scale read off a count (decisions 41, 44). One cell per process, one JSON line out. Numbers
# to the log only (the rule).
# Usage: sweep7.py <name> <forms> <nonterminals> <resolution> <width> <layers> <length> <tables> <train_radius_grains> <steps_mean> <switch_mean> <corpus_size> <read_radii: a,b,free> <read_size> <cap> <seed> [lr=0.1] [patience=3] [minimum=300] [lean=0.75]
import json, math, random, sys, time
from fractions import Fraction

from cocoonml.array_attention import ArrayAttention
from cocoonml.corpus import Ball, bin_windows, form_losses_by_rank, form_losses_by_rank_with_error, order_control, probe_rows, produce, whole_rank, window_distances, window_truth_distances, windows
from cocoonml.harness import train_until_plateau
from cocoonml.probe import fit, r_squared
from cocoonml.schema import generate

name = sys.argv[1]
forms, nts, res, width, layers, length, n_tables, train_radius_g = map(int, sys.argv[2:10])
steps_mean, switch_mean = float(sys.argv[10]), float(sys.argv[11])
corpus_size = int(sys.argv[12])
read_radii = [None if r == "free" else int(r) for r in sys.argv[13].split(",")]
read_size, cap, seed = int(sys.argv[14]), int(sys.argv[15]), int(sys.argv[16])
LR = float(sys.argv[17]) if len(sys.argv) > 17 else 0.1
PATIENCE = int(sys.argv[18]) if len(sys.argv) > 18 else 3
MINIMUM = int(sys.argv[19]) if len(sys.argv) > 19 else 300
LEAN = Fraction(sys.argv[20]) if len(sys.argv) > 20 else Fraction(3, 4)
SEP = forms + 1
BINS = 3
t0 = time.time()

tables = [generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=10 * seed + i, graded=True) for i in range(n_tables)]
bound = Fraction(nts, 2 * res)
grain = Fraction(1, res)
train_radius = train_radius_g * grain if train_radius_g > 0 else None
rng = random.Random(seed)
corpus = produce(tables, rng, bound, LEAN, train_radius, corpus_size, steps_mean, switch_mean)
ws = windows(corpus, SEP, length)
whole = whole_rank(ws)
held = produce(tables, random.Random(seed + 999), bound, LEAN, train_radius, max(16, corpus_size // 4), steps_mean, switch_mean)
hws = windows(held, SEP, length)
model = ArrayAttention(vocab=forms + 2, width=width, length=length, seed=seed, separator=SEP, layers=layers, scale="derived")  # the initialisation that reads off every count (decisions 41)
evaluate = lambda: sum(model.loss(t, g) for t, g, _ in hws[:16]) / len(hws[:16])
losses, evaluations = train_until_plateau(model, lambda: [(t, g) for t, g, _ in rng.sample(ws, min(4, len(ws)))], LR, window=50, tolerance=0.01, cap=cap, evaluate=evaluate, patience=PATIENCE, minimum=MINIMUM)
ball = Ball(state for walk in corpus.walks for state in walk)  # distinct states, distances cached (the readers' cost, 2026-10-09)
training_forms = [a.forms for a in corpus.artifacts]  # every training string, no provenance: the reader's side


def curves(window_list, rng_):
    w = min(whole, whole_rank(window_list)) if window_list else 0
    return {"windows": len(window_list), "whole": w, "form_by_rank": form_losses_by_rank(model, window_list, SEP, w), "form_by_rank_error": form_losses_by_rank_with_error(model, window_list, SEP, w), "order_control_by_rank": order_control(model, window_list, SEP, rng_, w)}


def read(window_list, source, rng_, base):
    """One reading's curves and its probe rows; `base` offsets the rows' window index so windows of
    different readings never share an index (the split in probe() is by window; decisions 45)."""
    estimated = window_distances(tables, window_list, training_forms)
    truth = window_truth_distances(window_list, source, ball)
    truth_np = window_truth_distances(window_list, source, ball, phases=False)
    known = [d for d in estimated if d is not None]
    out = curves(window_list, rng_)
    out["estimated_distance"] = sum(known) / len(known) if known else None
    out["estimated_known"] = len(known)
    out["truth_distance"] = sum(truth) / len(truth) if truth else None
    out["truth_distance_no_phase"] = sum(truth_np) / len(truth_np) if truth_np else None
    out["by_truth_distance"] = [{"distance": d, **curves(part, rng_)} for d, part in bin_windows(window_list, truth_np, BINS)]
    out["by_estimated_distance"] = [{"distance": d, **curves(part, rng_)} for d, part in bin_windows(window_list, estimated, BINS)]
    return out, [(x, d, k, w + base) for x, d, k, w in probe_rows(model, window_list, source, ball, SEP, phases=False)], base + len(window_list)


same, rows, base = read(ws[: max(4, len(ws) // 4)], corpus, random.Random(seed + 1), 0)  # in-sample: the training windows themselves
held_reading, more, base = read(hws, held, random.Random(seed + 3), base)  # fresh windows at the training radius
rows += more
points = {}
for r in read_radii:
    radius = None if r is None else r * grain
    # radius 0 is the stationary reading at the origins (no steps), not a free walk (a zero radius would turn the pull off)
    reading = produce(tables, random.Random(1000 * seed + (-1 if r is None else r)), bound, LEAN, radius, read_size, 0.0 if r == 0 else steps_mean, switch_mean)
    rws = windows(reading, SEP, length)
    points["free" if r is None else str(r)], more, base = read(rws, reading, random.Random(seed + 2), base)
    rows += more


def probe(rows, target):
    """r² on the windows held out of the fit: the split is by window, so an artifact's window-mates never sit on both sides."""
    window_ids = sorted({w for _, _, _, w in rows})
    random.Random(0).shuffle(window_ids)
    train_ids = set(window_ids[: len(window_ids) // 2])
    train = [(x, y) for x, y, w in ((row[0], row[target], row[3]) for row in rows) if w in train_ids]
    test = [(x, y) for x, y, w in ((row[0], row[target], row[3]) for row in rows) if w not in train_ids]
    if len(train) < width + 2 or len(test) < 2:
        return None
    wts, b = fit([x for x, _ in train], [y for _, y in train], ridge=1e-2)
    return r_squared(wts, b, [x for x, _ in test], [y for _, y in test])


print(json.dumps({
    "run": name, "args": sys.argv[1:], "seconds": round(time.time() - t0, 1), "tables": n_tables, "layers": layers, "width": width, "scale": "derived", "lr": LR, "ball_states": len(ball),
    "train_steps": len(losses), "hit_cap": len(losses) >= cap, "diverged": not math.isfinite(losses[-1]), "evaluations": evaluations, "train_first": losses[0], "train_last": losses[-1],
    "corpus_artifacts": len(corpus), "corpus_windows": len(ws), "whole": whole, "same": same, "held": held_reading, "points": points,
    "probe_r2_truth_distance": probe(rows, 1), "probe_r2_rank": probe(rows, 2),
}))
