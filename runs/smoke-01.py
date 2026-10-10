# Smoke run 01 of the apparatus (2026-10-08). Numbers go to the log, not to Izzy (the rule).
import json, time, random
from fractions import Fraction
from cocoonml.schema import generate, delta
from cocoonml.harness import run
t0 = time.time()
table = generate(forms=4, cuts=2, nonterminals=3, resolution=8, seed=21)
train_losses, same, after, moved = run(
    table, model_seed=2, width=4, length=12, train_steps=150, batch_size=4, per_context=3,
    lr=0.3, drift_steps=5, bound=Fraction(1, 2), lean=Fraction(3, 4), eval_count=20, seed=9,
)
out = {
    "run": "smoke-01", "seconds": round(time.time() - t0, 1),
    "train_first": train_losses[0], "train_last": train_losses[-1],
    "curve_same": same, "curve_drifted": after, "delta": str(delta(table, moved)),
}
print(json.dumps(out))
