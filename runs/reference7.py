# The analytical references of decisions 54 for a sweep-07 ensemble's held set: the marginal level
# (the unigram predictor with the training windows' token counts) and the Bayes floor (the
# predictor that knows the generator; exact with no steps, a particle filter with them), scored like
# the model over every position of each of the sixteen held windows runs/sweep7.py evaluates on.
# The held set and the training windows are reproduced from the ensemble's arguments and seeds
# exactly as sweep7.py produces them (seed + 999 for the held corpus). Two particle counts are run so
# the estimate's movement between them is on the record (the row's Fails-if). Numbers to the log.
# Usage: reference7.py <forms> <nonterminals> <resolution> <length> <tables> <train_radius_grains> <steps_mean> <switch_mean> <corpus_size> <seed> [lean=3/4] [particles=200,800] [particle_seed=11]
import random
import sys
import time
from fractions import Fraction

from cocoonml.corpus import produce, windows
from cocoonml.reference import bayes_losses, marginal_losses, token_counts
from cocoonml.schema import generate

if __name__ == "__main__":
    if len(sys.argv) < 11:
        sys.exit("usage: reference7.py <forms> <nonterminals> <resolution> <length> <tables> <train_radius_grains> <steps_mean> <switch_mean> <corpus_size> <seed> [lean=3/4] [particles=200,800] [particle_seed=11]")
    forms, nts, res, length, n_tables, train_radius_g = map(int, sys.argv[1:7])
    steps_mean, switch_mean = float(sys.argv[7]), float(sys.argv[8])
    corpus_size, seed = int(sys.argv[9]), int(sys.argv[10])
    lean = Fraction(sys.argv[11]) if len(sys.argv) > 11 else Fraction(3, 4)
    counts_of_particles = [int(x) for x in sys.argv[12].split(",")] if len(sys.argv) > 12 else [200, 800]
    particle_seed = int(sys.argv[13]) if len(sys.argv) > 13 else 11
    sep = forms + 1
    bound = Fraction(nts, 2 * res)
    grain = Fraction(1, res)
    radius = train_radius_g * grain if train_radius_g > 0 else None
    t0 = time.time()
    tables = [generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=10 * seed + i, graded=True) for i in range(n_tables)]
    training = produce(tables, random.Random(seed), bound, lean, radius, corpus_size, steps_mean, switch_mean)
    ws = windows(training, sep, length)
    held = produce(tables, random.Random(seed + 999), bound, lean, radius, max(16, corpus_size // 4), steps_mean, switch_mean)
    hws = windows(held, sep, length)[:16]
    counts = token_counts(ws)
    print(f"ensemble: forms {forms}, nonterminals {nts}, resolution {res}, length {length}, tables {n_tables}, radius {train_radius_g} grains, steps {steps_mean}, switches {switch_mean}, corpus {corpus_size} artifacts ({len(ws)} windows), seed {seed}; held set {len(hws)} windows; produced in {time.time() - t0:.1f} s", flush=True)
    marginal = [sum(marginal_losses(g, counts)) / len(t) for t, g, _ in hws]
    print(f"marginal level (unigram, training counts): mean {sum(marginal) / len(marginal):.4f}; per window {[round(x, 4) for x in marginal]}", flush=True)
    if steps_mean <= 0 or radius is None and steps_mean <= 0:
        exact = [sum(bayes_losses(t, g, sep, tables, switch_mean)) / len(t) for t, g, _ in hws]
        print(f"bayes floor (exact, no steps): mean {sum(exact) / len(exact):.4f}; per window {[round(x, 4) for x in exact]}", flush=True)
    else:
        for k in counts_of_particles:
            t1 = time.time()
            rng = random.Random(particle_seed)
            floor = [sum(bayes_losses(t, g, sep, tables, switch_mean, steps_mean=steps_mean, bound=bound, lean=lean, radius=radius, particles=k, rng=rng)) / len(t) for t, g, _ in hws]
            print(f"bayes floor (particle filter, {k} particles, seed {particle_seed}): mean {sum(floor) / len(floor):.4f}; per window {[round(x, 4) for x in floor]}; {time.time() - t1:.0f} s", flush=True)
    stationary = [sum(bayes_losses(t, g, sep, tables, switch_mean)) / len(t) for t, g, _ in hws]
    print(f"for scale, the stationary tables' exact predictor on the same windows (misspecified: it ignores the walk): mean {sum(stationary) / len(stationary):.4f}", flush=True)
    print(f"done in {time.time() - t0:.0f} s")
