# Sweep 02 (2026-10-08), the design after round one: step bound at the grain, training to a plateau,
# n_eff against the drifted table and foreignness against the old, delta_magnitude as the probe
# target, form by rank, meaning with whole artifacts only, a hundred windows per depth.
# Usage: sweep2.py <name> <forms> <nonterminals> <resolution> <width> <length> <per_context> <train_steps> <windows>
import json, sys, time, random
from fractions import Fraction
from cocoonml.schema import generate, step, delta_magnitude, n_eff, draw
from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, meaning_loss_at_separators, form_loss_by_rank, pack
from cocoonml.probe import fit, r_squared

name, forms, nts, res, width, length, per, steps, windows = sys.argv[1], *map(int, sys.argv[2:])
SEP, BASE = forms + 1, forms + 2
DRIFTS = [0, 2, 4, 8, 16]
t0 = time.time(); rows = []
for tseed in (1, 2, 3):
    table = generate(forms=forms, cuts=2, nonterminals=nts, resolution=res, seed=tseed)
    bound = Fraction(nts, res)  # per node: one grain per step at most
    for mseed in (1, 2, 3):
        rng = random.Random(100 * tseed + mseed)
        model = Attention(vocab=BASE + 3 * length, width=width, length=length, seed=mseed)
        train = [model.train_step(contexts_meaning(table, 4, per, SEP, length, BASE, rng), 0.3) for _ in range(steps)]
        xs, ys, per_drift = [], [], {}
        for d in DRIFTS:
            moved = table
            for _ in range(d):
                moved = step(moved, rng, bound, Fraction(3, 4))
            dl = delta_magnitude(table, moved)
            ws = [[draw(moved, rng)[0] for _ in range(per)] for _ in range(windows)]
            neff_moved = sum(n_eff(moved, w, 4)[0] for w in ws) / windows
            foreign = sum(n_eff(table, w, 4)[0] for w in ws) / windows
            form_curve = form_loss_by_rank(model, moved, windows // 2, per, SEP, rng)
            meaning_curve = meaning_loss_at_separators(model, moved, windows // 2, per, SEP, length, BASE, rng)
            for w in ws:
                tokens, _ = pack(w, SEP, length)
                acts = model.probe(tokens)
                last_sep = max(i for i, t in enumerate(tokens) if t == SEP)
                xs.append(acts[last_sep]); ys.append(dl)
            per_drift[d] = {"delta_magnitude": dl, "n_eff_moved": neff_moved, "foreignness": foreign, "form_by_rank": form_curve, "meaning_by_rank": meaning_curve}
        idx = list(range(len(xs))); random.Random(0).shuffle(idx); half = len(idx) // 2
        w, b = fit([xs[i] for i in idx[:half]], [ys[i] for i in idx[:half]], ridge=1e-2)
        r2 = r_squared(w, b, [xs[i] for i in idx[half:]], [ys[i] for i in idx[half:]])
        rows.append({"table": tseed, "model": mseed, "train_first": train[0], "train_mid": train[len(train)//2], "train_last": train[-1], "probe_r2_heldout": r2, "per_drift": per_drift})
        print(json.dumps({"progress": f"{name} table {tseed} model {mseed}", "seconds": round(time.time() - t0, 1)}), flush=True)
print(json.dumps({"run": name, "args": sys.argv[1:], "seconds": round(time.time() - t0, 1), "rows": rows}))
