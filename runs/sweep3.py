# Sweep 03 (2026-10-08), the design after round one against sweeps 02: the step reflects at the ends
# and is pulled toward the origin within a radius (no freeze); the identification error beside
# n_eff; losses reported beside the exact floors (excess = loss - floor of the table read);
# contexts that fill, for training and reading; curves only at ranks every context reaches; the
# probe's target continuous: the drift each window shows against the training table. One drift
# walk per table, read by three models. Numbers go to the log only (the rule).
# Usage: sweep3.py <name> <forms> <nonterminals> <resolution> <width> <length> <train_steps> <windows> <drift_seed> <radius_grains>
import json, random, sys, time
from fractions import Fraction

from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, filled, form_loss_by_rank, meaning_loss_at_separators
from cocoonml.probe import fit, r_squared
from cocoonml.schema import delta_magnitude, entropy_floor_form, entropy_floor_meaning, generate, identification_error, n_eff, step

name, forms, nts, res, width, length, steps, windows, drift_seed, radius_grains = sys.argv[1], *map(int, sys.argv[2:])
SEP, BASE = forms + 1, forms + 2
DEPTHS = [0, 2, 4, 8, 16, 32]
LEAN = Fraction(3, 4)


def strings_of(tokens, n):
    """The first n artifacts of a packed context, as strings."""
    segments, current = [], []
    for t in tokens:
        if t == SEP:
            segments.append(tuple(current))
            current = []
        else:
            current.append(t)
    return segments[:n]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


t0 = time.time()
rows = []
for tseed in (1, 2, 3):
    table = generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=tseed)
    bound = Fraction(nts, 2 * res)  # each node moves one grain with probability one half per step
    radius = Fraction(radius_grains, res)
    drift_rng = random.Random(1000 * tseed + drift_seed)
    moved_by_depth, moved, at = {}, table, 0
    for d in DEPTHS:
        while at < d:
            moved = step(moved, drift_rng, bound, LEAN, centre=table, radius=radius)
            at += 1
        moved_by_depth[d] = moved
    floors_by_depth = {d: {"form": entropy_floor_form(m), "meaning": entropy_floor_meaning(m)} for d, m in moved_by_depth.items()}
    for mseed in (1, 2, 3):
        rng = random.Random(100 * tseed + mseed)
        model = Attention(vocab=BASE + 3 * length, width=width, length=length, seed=mseed)
        train = [model.train_step(contexts_meaning(table, 4, 0, SEP, length, BASE, rng, fill=True), 0.3) for _ in range(steps)]
        xs, ys, per_depth = [], [], {}
        for d in DEPTHS:
            m = moved_by_depth[d]
            ws = [filled(m, SEP, length, rng, BASE) for _ in range(windows)]
            strings = [strings_of(tokens, n) for tokens, _, n in ws]
            ident = [identification_error(m, s) for s in strings]
            shown = [identification_error(table, s)[0] for s in strings]
            form_curve = form_loss_by_rank(model, m, windows // 2, 0, SEP, rng, fill=True)
            meaning_curve = meaning_loss_at_separators(model, m, windows // 2, 0, SEP, length, BASE, rng, fill=True)
            for (tokens, _, n), wd in zip(ws, shown):
                if wd is None or n == 0:
                    continue
                seps = [i for i, t in enumerate(tokens) if t == SEP]
                xs.append(model.probe(tokens)[seps[n - 1]])
                ys.append(wd)
            per_depth[d] = {
                "delta_magnitude": delta_magnitude(table, m),
                "floors": floors_by_depth[d],
                "n_eff_moved": mean(n_eff(m, s, 4)[0] for s in strings),
                "foreignness": mean(n_eff(table, s, 4)[0] for s in strings),
                "identification_error": mean(e for e, _, _ in ident),
                "coverage": mean(c for _, c, _ in ident),
                "window_drift_shown": mean(shown),
                "artifacts_per_window": mean(n for _, _, n in ws),
                "form_by_rank": form_curve,
                "form_excess_by_rank": [x - floors_by_depth[d]["form"] for x in form_curve],
                "meaning_by_rank": meaning_curve,
                "meaning_excess_by_rank": [x - floors_by_depth[d]["meaning"] for x in meaning_curve],
            }
        idx = list(range(len(xs)))
        random.Random(0).shuffle(idx)
        half = len(idx) // 2
        r2 = None
        if half >= width + 2:
            w, b = fit([xs[i] for i in idx[:half]], [ys[i] for i in idx[:half]], ridge=1e-2)
            r2 = r_squared(w, b, [xs[i] for i in idx[half:]], [ys[i] for i in idx[half:]])
        rows.append({"table": tseed, "model": mseed, "train_first": train[0], "train_mid": train[len(train) // 2], "train_last": train[-1],
                     "train_floor_meaning": floors_by_depth[0]["meaning"], "probe_r2_heldout": r2, "probe_windows": len(xs), "per_depth": per_depth})
        print(json.dumps({"progress": f"{name} table {tseed} model {mseed}", "seconds": round(time.time() - t0, 1)}), flush=True)
print(json.dumps({"run": name, "args": sys.argv[1:], "seconds": round(time.time() - t0, 1), "rows": rows}))
