# Sweep 07 (2026-10-09, the complete system's first script; not yet run at the derived size): the
# corpus layer in place of the harness's contexts. Several graded tables in play, each walking within
# the training radius; a finite corpus produced in order with drawn steps and switches; windows as
# stretches of production (states succeed, tables mix, no extension token); training samples windows
# in no order and ends on the held-out plateau rule; readings on corpora walked to each read radius
# (grains; "free" for an unbounded walk): the form curve by rank, the order control, the reader's
# distance (shares a window pins against the corpus's) with the truth's distance from the ball as its
# check, and a probe from the separator vectors to the truth's distance. The lean is an argument
# (swept, not a constant). One cell per process, one JSON line out. Numbers to the log only (the rule).
# Usage: sweep7.py <name> <forms> <nonterminals> <resolution> <width> <layers> <length> <tables> <train_radius_grains> <steps_mean> <switch_mean> <corpus_size> <read_radii: a,b,free> <read_size> <cap> <seed> [lr=0.1] [patience=3] [minimum=300] [lean=0.75]
import json, random, sys, time
from fractions import Fraction

from cocoonml.attention import Attention
from cocoonml.corpus import distance_from_ball, estimated_distance, form_losses_by_rank, order_control, probe_rows, produce, whole_rank, windows
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
hws = windows(held, SEP, length)[:16]
model = Attention(vocab=forms + 2, width=width, length=length, seed=seed, separator=SEP, layers=layers)
evaluate = lambda: sum(model.loss(t, g).value for t, g, _ in hws) / len(hws)
losses, evaluations = train_until_plateau(model, lambda: [(t, g) for t, g, _ in rng.sample(ws, min(4, len(ws)))], LR, window=50, tolerance=0.01, cap=cap, evaluate=evaluate, patience=PATIENCE, minimum=MINIMUM)
ball = [state for walk in corpus.walks for state in walk]
training_forms = [[a.forms for a in corpus.artifacts if a.table == i] for i in range(n_tables)]


def read(window_list, source, rng_):
    w = min(whole, whole_rank(window_list)) if window_list else 0
    per_table = []
    for i, shape in enumerate(tables):
        reading_forms = [a.forms for _, _, used in window_list for a in used if a.table == i]
        d = estimated_distance(shape, reading_forms, training_forms[i]) if reading_forms and training_forms[i] else None
        if d is not None:
            per_table.append(d)
    truth = [distance_from_ball(source.walks[a.table][a.state], ball) for _, _, used in window_list for a in used]
    return {
        "windows": len(window_list),
        "whole": w,
        "form_by_rank": form_losses_by_rank(model, window_list, SEP, w),
        "order_control_by_rank": order_control(model, window_list, SEP, rng_, w),
        "estimated_distance": sum(per_table) / len(per_table) if per_table else None,
        "truth_distance": sum(truth) / len(truth) if truth else None,
    }, probe_rows(model, window_list, source, ball, SEP)


same, rows = read(ws[: max(4, len(ws) // 4)], corpus, random.Random(seed + 1))
points = {}
for r in read_radii:
    radius = None if r is None else r * grain
    # radius 0 is the stationary reading at the origins (no steps), not a free walk (a zero radius would turn the pull off)
    reading = produce(tables, random.Random(1000 * seed + (-1 if r is None else r)), bound, LEAN, radius, read_size, 0.0 if r == 0 else steps_mean, switch_mean)
    rws = windows(reading, SEP, length)
    points["free" if r is None else str(r)], more = read(rws, reading, random.Random(seed + 2))
    rows += more
r2 = None
idx = list(range(len(rows)))
random.Random(0).shuffle(idx)
half = len(idx) // 2
if half >= width + 2:
    w, b = fit([rows[i][0] for i in idx[:half]], [rows[i][1] for i in idx[:half]], ridge=1e-2)
    r2 = r_squared(w, b, [rows[i][0] for i in idx[half:]], [rows[i][1] for i in idx[half:]])
print(json.dumps({
    "run": name, "args": sys.argv[1:], "seconds": round(time.time() - t0, 1), "tables": n_tables, "layers": layers, "width": width,
    "train_steps": len(losses), "hit_cap": len(losses) >= cap, "evaluations": evaluations, "train_first": losses[0], "train_last": losses[-1],
    "corpus_artifacts": len(corpus), "corpus_windows": len(ws), "whole": whole, "same": same, "points": points, "probe_r2_truth_distance": r2,
}))
