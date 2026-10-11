# Sweep 08 (2026-10-11, decisions 55; docs/literature.md entry 1): the Markov-chain world through the same model and
# readers as sweep 07, so that what differs is the data. A table is a chain drawn from a Dirichlet prior and quantised
# to the grid (cocoonml/markov.py); an artifact one run of the chain to its end; the production (switching, steps within
# the training radius, packing) is corpus.produce with the world's draw and step. Every reading of sweep7.py is taken
# here with the chain world's ball and distance; the JSON line adds decisions 54's two references on the sixteen held
# windows, the marginal level and the Bayes floor (exact with no steps, a particle filter with them), so the held
# trajectory reads as their KL-to-optimum curve. The walk's bound is (states + 1) / (2 · resolution), the schema's
# nonterminals / (2 · resolution) with the rows in place of the nodes: one grain per two rows per step on average, a
# choice marked in the row. One cell per process, one JSON line out. Numbers to the log only (the rule).
# Usage: sweep8.py <name> <states> <resolution> <width> <layers> <length> <chains> <train_radius_grains> <steps_mean> <switch_mean> <corpus_size> <read_radii: a,b,free> <read_size> <cap> <seed> [lr=0.1] [patience=3] [minimum=300] [alpha=1.0] [end_alpha=0.5] [mode=plateau|cap] [copies_dir] [particles=200]
import json, math, os, random, sys, time
from fractions import Fraction

import numpy as np

from cocoonml.array_attention import ArrayAttention
from cocoonml.circuits import circuit_scores
from cocoonml.corpus import bin_windows, form_losses_by_rank, form_losses_by_rank_with_error, order_control, probe_rows, produce, whole_rank, window_truth_distances, windows
from cocoonml.harness import train_to_cap, train_until_plateau
from cocoonml.markov import MarkovBall, draw, generate, step, window_distances
from cocoonml.probe import fit, r_squared
from cocoonml.reference import MarkovWorld, bayes_losses, marginal_losses, token_counts
from cocoonml.rules import ENVELOPE, onset

name = sys.argv[1]
states, res, width, layers, length, n_tables, train_radius_g = map(int, sys.argv[2:9])
steps_mean, switch_mean = float(sys.argv[9]), float(sys.argv[10])
corpus_size = int(sys.argv[11])
read_radii = [None if r == "free" else int(r) for r in sys.argv[12].split(",")]
read_size, cap, seed = int(sys.argv[13]), int(sys.argv[14]), int(sys.argv[15])
LR = float(sys.argv[16]) if len(sys.argv) > 16 else 0.1
PATIENCE = int(sys.argv[17]) if len(sys.argv) > 17 else 3
MINIMUM = int(sys.argv[18]) if len(sys.argv) > 18 else 300
ALPHA = float(sys.argv[19]) if len(sys.argv) > 19 else 1.0
END_ALPHA = float(sys.argv[20]) if len(sys.argv) > 20 else 0.5
MODE = sys.argv[21] if len(sys.argv) > 21 else "plateau"
COPIES_DIR = sys.argv[22] if len(sys.argv) > 22 else None
PARTICLES = int(sys.argv[23]) if len(sys.argv) > 23 else 200
LEAN = None  # no meaning in this world
forms = states  # the states are the forms 1..k
if MODE not in ("plateau", "cap"):
    sys.exit(f"mode {MODE!r}: plateau or cap")
WINDOW = 50
SEP = forms + 1
BINS = 3
t0 = time.time()

tables = [generate(states=states, resolution=res, seed=10 * seed + i, alpha=ALPHA, end_alpha=END_ALPHA) for i in range(n_tables)]
bound = Fraction(states + 1, 2 * res)
grain = Fraction(1, res)
train_radius = train_radius_g * grain if train_radius_g > 0 else None
rng = random.Random(seed)
corpus = produce(tables, rng, bound, LEAN, train_radius, corpus_size, steps_mean, switch_mean, draw=draw, step=step)
ws = windows(corpus, SEP, length)
whole = whole_rank(ws)
held = produce(tables, random.Random(seed + 999), bound, LEAN, train_radius, max(16, corpus_size // 4), steps_mean, switch_mean, draw=draw, step=step)
hws = windows(held, SEP, length)
model = ArrayAttention(vocab=forms + 2, width=width, length=length, seed=seed, separator=SEP, layers=layers, scale="derived")  # the initialisation that reads off every count (decisions 41)
evaluate = lambda: sum(model.loss(t, g) for t, g, _ in hws[:16]) / len(hws[:16])
make_batch = lambda: [(t, g) for t, g, _ in rng.sample(ws, min(4, len(ws)))]
if MODE == "cap":
    record = train_to_cap(model, make_batch, LR, cap, evaluate, lambda ev: onset(ev, ENVELOPE, window=WINDOW), window=WINDOW)
    losses, evaluations = record["losses"], record["evaluations"]
else:
    record = None
    losses, evaluations = train_until_plateau(model, make_batch, LR, window=WINDOW, tolerance=0.01, cap=cap, evaluate=evaluate, patience=PATIENCE, minimum=MINIMUM)
ball = MarkovBall(state for walk in corpus.walks for state in walk)  # distinct states, distances cached
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


def read_everything():
    """Every reading of the model as it stands: the in-sample windows, the held windows, each read radius, and the probe rows."""
    same, rows, base = read(ws[: max(4, len(ws) // 4)], corpus, random.Random(seed + 1), 0)  # in-sample: the training windows themselves
    held_reading, more, base = read(hws, held, random.Random(seed + 3), base)  # fresh windows at the training radius
    rows += more
    points = {}
    for r in read_radii:
        radius = None if r is None else r * grain
        # radius 0 is the stationary reading at the origins (no steps), not a free walk (a zero radius would turn the pull off)
        reading = produce(tables, random.Random(1000 * seed + (-1 if r is None else r)), bound, LEAN, radius, read_size, 0.0 if r == 0 else steps_mean, switch_mean, draw=draw, step=step)
        rws = windows(reading, SEP, length)
        points["free" if r is None else str(r)], more, base = read(rws, reading, random.Random(seed + 2), base)
        rows += more
    circuits = circuit_scores(model, [t for t, _, _ in hws[:16]], exclude={SEP})  # the two circuit scores per layer on the held windows (docs/literature.md, entries 2 and 3)
    return {"same": same, "held": held_reading, "points": points, "probe_r2_truth_distance": probe(rows, 1), "probe_r2_rank": probe(rows, 2), "circuits": circuits}


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


line = {
    "run": name, "args": sys.argv[1:], "seconds": None, "tables": n_tables, "layers": layers, "width": width, "scale": "derived", "lr": LR, "ball_states": len(ball),
    "train_steps": len(losses), "hit_cap": len(losses) >= cap, "diverged": not math.isfinite(losses[-1]), "evaluations": evaluations, "train_first": losses[0], "train_last": losses[-1],
    "corpus_artifacts": len(corpus), "corpus_windows": len(ws), "whole": whole, "mode": MODE,
}
if record is None:
    line.update(read_everything())
else:
    copies = record["copies"]
    line.update(escape_step=record["escape_step"], onset_step=record["onset_step"], batch_losses=losses, diverged=line["diverged"] or record["escape_step"] is not None)
    line["copies"] = {}
    for copy_name in ("marginal", "best", "last"):
        copy = copies[copy_name]
        if copy is None:
            line["copies"][copy_name] = None
            continue
        entry = {"step": copy.step, "held_loss": copy.held}  # "held" below is the copy's held reading (read_everything); the loss has its own key
        if COPIES_DIR:
            os.makedirs(COPIES_DIR, exist_ok=True)
            entry["file"] = os.path.join(COPIES_DIR, f"{name}-{copy_name}.npz")
            np.savez(entry["file"], *copy.blocks)
        if copy_name != "last":  # the last copy is kept and not read by the row (decisions 53, second form (vi))
            copy.load_into(model)
            entry.update(read_everything())
        line["copies"][copy_name] = entry
    read_copy = "best" if copies["best"] is not None else "marginal"
    line["read_copy"] = read_copy
    if line["copies"][read_copy] is not None:
        line.update({key: line["copies"][read_copy][key] for key in ("same", "held", "points", "probe_r2_truth_distance", "probe_r2_rank", "circuits")})
# decisions 54's references on the evaluate set (hws[:16]): the marginal level from the training windows' counts and
# the Bayes floor, exact with no steps and by the particle filter with them
counts = token_counts(ws)
world = MarkovWorld()
marginal = [sum(marginal_losses(g, counts)) / len(t) for t, g, _ in hws[:16]]
if steps_mean <= 0:
    floor = [sum(bayes_losses(t, g, SEP, tables, switch_mean, world=world)) / len(t) for t, g, _ in hws[:16]]
    how = "exact"
else:
    prng = random.Random(seed + 11)
    floor = [sum(bayes_losses(t, g, SEP, tables, switch_mean, steps_mean=steps_mean, bound=bound, lean=LEAN, radius=train_radius, particles=PARTICLES, rng=prng, world=world)) / len(t) for t, g, _ in hws[:16]]
    how = PARTICLES
line["reference"] = {"marginal": sum(marginal) / len(marginal), "floor": sum(floor) / len(floor), "floor_estimate": how, "marginal_per_window": marginal, "floor_per_window": floor}
line.update(world="markov", states=states, resolution=res, alpha=ALPHA, end_alpha=END_ALPHA)
line["seconds"] = round(time.time() - t0, 1)
print(json.dumps(line))
