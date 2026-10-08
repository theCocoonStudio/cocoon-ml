# Sweep 01 (2026-10-08): three table seeds x three model seeds; drift steps 0,1,2,4,8;
# per window: n_eff, the form curve by position, the meaning curve by separator rank, and a
# linear probe from the last separator's attended vector to the window's delta (fit on half the
# windows, r-squared on the other half). Numbers to the log only; the rule applies.
import json, time, random
from fractions import Fraction
from cocoonml.schema import generate, step, delta, n_eff, draw
from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, meaning_loss_at_separators, loss_by_position
from cocoonml.probe import fit, r_squared

SEP, BASE, LENGTH, PER = 5, 10, 14, 3
DRIFTS = [0, 1, 2, 4, 8]
t0 = time.time()
rows = []
for tseed in (1, 2, 3):
    table = generate(forms=4, cuts=2, nonterminals=3, resolution=8, seed=tseed)
    for mseed in (1, 2, 3):
        rng = random.Random(100 * tseed + mseed)
        model = Attention(vocab=BASE + 12, width=4, length=LENGTH, seed=mseed)
        train = [model.train_step(contexts_meaning(table, 4, PER, SEP, LENGTH, BASE, rng), 0.3) for _ in range(150)]
        xs, ys, per_drift = [], [], {}
        for d in DRIFTS:
            moved = table
            for _ in range(d):
                moved = step(moved, rng, Fraction(1, 2), Fraction(3, 4))
            dl = float(delta(table, moved))
            windows = [[draw(moved, rng)[0] for _ in range(PER)] for _ in range(12)]
            neffs = [n_eff(table, w, 4)[0] for w in windows]
            form_curve = loss_by_position(model, moved, 12, PER, SEP, rng)
            meaning_curve = meaning_loss_at_separators(model, moved, 12, PER, SEP, LENGTH, BASE, rng)
            # probe features: the attended vector at the last separator of each window's context
            for w in windows:
                from cocoonml.harness import pack
                tokens, _ = pack(w, SEP, LENGTH)
                rows_act = model.probe(tokens)
                last_sep = max(i for i, t in enumerate(tokens) if t == SEP)
                xs.append(rows_act[last_sep]); ys.append(dl)
            per_drift[d] = {"delta": dl, "n_eff_mean": sum(neffs) / len(neffs), "form_curve": form_curve, "meaning_curve": meaning_curve}
        half = len(xs) // 2
        idx = list(range(len(xs))); random.Random(0).shuffle(idx)
        tr, te = idx[:half], idx[half:]
        w, b = fit([xs[i] for i in tr], [ys[i] for i in tr], ridge=1e-2)
        r2 = r_squared(w, b, [xs[i] for i in te], [ys[i] for i in te])
        rows.append({"table": tseed, "model": mseed, "train_first": train[0], "train_last": train[-1], "probe_r2_heldout": r2, "per_drift": per_drift})
        print(json.dumps({"progress": f"table {tseed} model {mseed} done", "seconds": round(time.time() - t0, 1)}), flush=True)
print(json.dumps({"run": "sweep-01", "seconds": round(time.time() - t0, 1), "rows": rows}))
