# Reference 8 (2026-10-18, the solo stretch): decisions 54's two references at every reading radius of a
# sweep-08 cell, on the same windows the model is read on, so that the unresolved share S of
# docs/proofs/general-model.md 6.4 can be read by rank at a distance: S(R, k, D) = ex_k(R | D) / ex_k(R0 | D),
# the model's excess over the Bayes floor as a share of the marginal reader's. The cell's JSON line gives
# the arguments; the world, the training corpus (the marginal's counts and the first artifact's length
# prior) and every reading corpus are regenerated from the seeds sweep8.py uses; a saved copy of the
# parameters is loaded into the model. Per reading radius, on the first <windows> windows: the model's
# form loss by rank, the marginal's by rank in two forms (the training windows' counts, R0 under the
# training law; the reading windows' own counts, R0 under the reading law, the one 4.3 names for S), the
# floor's by rank (exact where the reading has no steps, the particle filter otherwise; the floor is the
# Bayes reader under the reading law, so it carries the sampling floor and the moving index), and S by
# rank with each marginal. One JSON line out. Numbers to the log only (the rule).
# Usage: reference8.py <cell_log> <copy: marginal|best|last> <windows> <particles> [radii: a,b,free]
import json
import random
import sys
import time
from fractions import Fraction

import numpy as np

from cocoonml.array_attention import ArrayAttention
from cocoonml.corpus import _by_rank, form_losses_by_rank, position_losses, produce, whole_rank, window_truth_distances, windows
from cocoonml.markov import MarkovBall, draw, generate, step
from cocoonml.reference import MarkovWorld, bayes_losses, leftover_distribution, marginal_losses, token_counts

log_path, copy_name, n_windows, particles = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(log_path) as f:
    cell = next(json.loads(ln) for ln in f if ln.startswith('{"run"'))
a = cell["args"]
name = a[0]
states, res, width, layers, length, n_tables, train_radius_g = map(int, a[1:8])
steps_mean, switch_mean = float(a[8]), float(a[9])
corpus_size, read_size, seed = int(a[10]), int(a[12]), int(a[14])
alpha = float(a[18]) if len(a) > 18 else 1.0
end_alpha = float(a[19]) if len(a) > 19 else 0.5
radii_arg = sys.argv[5] if len(sys.argv) > 5 else a[11]
read_radii = [None if r == "free" else int(r) for r in radii_arg.split(",")]
copy = cell["copies"][copy_name]
if copy is None:
    sys.exit(f"{name}: no {copy_name} copy in the line")
t0 = time.time()

forms = states
SEP = forms + 1
tables = [generate(states=states, resolution=res, seed=10 * seed + i, alpha=alpha, end_alpha=end_alpha) for i in range(n_tables)]
bound = Fraction(states + 1, 2 * res)
grain = Fraction(1, res)
train_radius = train_radius_g * grain if train_radius_g > 0 else None
rng = random.Random(seed)
corpus = produce(tables, rng, bound, None, train_radius, corpus_size, steps_mean, switch_mean, draw=draw, step=step)
ws = windows(corpus, SEP, length)
ball = MarkovBall(state for walk in corpus.walks for state in walk)
counts_train = token_counts(ws)
leftovers = leftover_distribution(ws, length)
world = MarkovWorld()

model = ArrayAttention(vocab=forms + 2, width=width, length=length, seed=seed, separator=SEP, layers=layers, scale="derived")
saved = np.load(copy["file"])
sources = [saved[f"arr_{i}"] for i in range(len(saved.files))]
targets = model.blocks()
if len(sources) != len(targets):
    sys.exit(f"{name}: the copy holds {len(sources)} blocks, the model {len(targets)}")
for target, source in zip(targets, sources):
    target[...] = source


def by_rank_sums(window_list, per_window_losses):
    """Sums and counts by rank of per-position losses given per window, over form positions only (corpus._by_rank)."""
    sums, counts = {}, {}
    for (tokens, targets_, _), losses in zip(window_list, per_window_losses):
        s, c = _by_rank(tokens, targets_, losses, SEP)
        for r in s:
            sums[r] = sums.get(r, 0.0) + s[r]
            counts[r] = counts.get(r, 0) + c[r]
    return sums, counts


def by_rank_means(window_list, per_window_losses, whole):
    """Means by rank, cut at `whole` (the rank every window reaches, the record's rule since sweep 02)."""
    sums, counts = by_rank_sums(window_list, per_window_losses)
    return [sums[r] / counts[r] for r in sorted(sums) if r < whole]


def by_rank_means_deep(window_list, per_window_losses):
    """Means by rank with no cut, and the count of form positions behind each: ranks past `whole` are
    read on the windows that reach them only, a selection the reader must weigh (the deep curve)."""
    sums, counts = by_rank_sums(window_list, per_window_losses)
    return [sums[r] / counts[r] for r in sorted(sums)], [counts[r] for r in sorted(sums)]


def share(model_curve, floor_curve, marginal_curve):
    """S by rank: the model's excess over the floor as a share of the marginal's; None where the marginal resolves nothing."""
    out = []
    for m, fl, mg in zip(model_curve, floor_curve, marginal_curve):
        denominator = mg - fl
        out.append((m - fl) / denominator if abs(denominator) > 1e-9 else None)
    return out


points = {}
for r in read_radii:
    radius = None if r is None else r * grain
    reading_steps = 0.0 if r == 0 else steps_mean
    reading = produce(tables, random.Random(1000 * seed + (-1 if r is None else r)), bound, None, radius, read_size, reading_steps, switch_mean, draw=draw, step=step)
    rws = windows(reading, SEP, length)
    subset = rws[:n_windows]
    whole = whole_rank(subset)
    model_curve = form_losses_by_rank(model, subset, SEP, whole)
    model_deep, deep_counts = by_rank_means_deep(subset, [position_losses(model, t, g) for t, g, _ in subset])
    counts_read = token_counts(rws)
    marginal_train = by_rank_means(subset, [marginal_losses(g, counts_train) for _, g, _ in subset], whole)
    marginal_read = by_rank_means(subset, [marginal_losses(g, counts_read) for _, g, _ in subset], whole)
    prng = random.Random(seed + 11)
    floors, kept, retries, skipped = [], [], 0, 0
    for t, g, _ in subset:
        if reading_steps <= 0:
            floors.append(bayes_losses(t, g, SEP, tables, switch_mean, world=world, leftovers=leftovers))
            kept.append((t, g, _))
            continue
        # the particle filter can find no particle with mass on a window far from the origins (every
        # particle's likelihood zero, "probability zero under the generator"): the window is retried once
        # with five times the particles and skipped if that fails too; the counts are recorded, since the
        # floor is then read on the windows the filter could evaluate, a selection
        for n_particles in (particles, 5 * particles):
            try:
                floors.append(bayes_losses(t, g, SEP, tables, switch_mean, steps_mean=reading_steps, bound=bound, lean=None, radius=radius, particles=n_particles, rng=prng, world=world, leftovers=leftovers))
                kept.append((t, g, _))
                break
            except ValueError:
                if n_particles == particles:
                    retries += 1
                else:
                    skipped += 1
    how = "exact" if reading_steps <= 0 else particles
    floor_curve = by_rank_means(kept, floors, whole)
    floor_deep, _ = by_rank_means_deep(kept, floors)
    marginal_read_deep, _ = by_rank_means_deep(subset, [marginal_losses(g, counts_read) for _, g, _ in subset])
    if skipped:  # the model's and the marginal's curves on the same kept windows, so S compares like with like
        model_curve = form_losses_by_rank(model, kept, SEP, whole)
        model_deep, deep_counts = by_rank_means_deep(kept, [position_losses(model, t, g) for t, g, _ in kept])
        marginal_train = by_rank_means(kept, [marginal_losses(g, counts_train) for _, g, _ in kept], whole)
        marginal_read = by_rank_means(kept, [marginal_losses(g, counts_read) for _, g, _ in kept], whole)
        marginal_read_deep, _ = by_rank_means_deep(kept, [marginal_losses(g, counts_read) for _, g, _ in kept])
    truth = window_truth_distances(subset, reading, ball, phases=False)
    points["free" if r is None else str(r)] = {
        "windows": len(subset), "of": len(rws), "whole": whole, "floor_windows": len(kept), "floor_retries": retries, "floor_skipped": skipped, "reading_states": len(MarkovBall(s for walk in reading.walks for s in walk)),
        "truth_distance": sum(truth) / len(truth) if truth else None,
        "model_by_rank": model_curve, "floor_by_rank": floor_curve, "floor_estimate": how,
        "marginal_train_by_rank": marginal_train, "marginal_read_by_rank": marginal_read,
        "S_by_rank": share(model_curve, floor_curve, marginal_read), "S_train_marginal_by_rank": share(model_curve, floor_curve, marginal_train),
        "deep": {"counts": deep_counts, "model_by_rank": model_deep, "floor_by_rank": floor_deep, "marginal_read_by_rank": marginal_read_deep, "S_by_rank": share(model_deep, floor_deep, marginal_read_deep)},
    }

print(json.dumps({"run": name, "copy": copy_name, "copy_step": copy["step"], "copy_held_loss": copy["held_loss"], "windows": n_windows, "particles": particles,
                  "ball_states": len(ball), "train_radius_grains": train_radius_g, "points": points, "seconds": round(time.time() - t0, 1)}))
