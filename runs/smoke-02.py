# Smoke run 02: the meaning harness end to end (2026-10-08). Numbers to the log only.
import json, time, random
from fractions import Fraction
from cocoonml.schema import generate, delta, step
from cocoonml.attention import Attention
from cocoonml.harness import contexts_meaning, meaning_loss_at_separators
t0 = time.time()
rng = random.Random(3)
table = generate(forms=4, cuts=2, nonterminals=3, resolution=8, seed=21)
sep, base, length, per = 5, 10, 14, 3
model = Attention(vocab=base + 12, width=4, length=length, seed=2)
losses = []
for _ in range(150):
    batch = contexts_meaning(table, 4, per, sep, length, base, rng)
    losses.append(model.train_step(batch, 0.3))
same = meaning_loss_at_separators(model, table, 20, per, sep, length, base, rng)
moved = table
for _ in range(5):
    moved = step(moved, rng, Fraction(1, 2), Fraction(3, 4))
after = meaning_loss_at_separators(model, moved, 20, per, sep, length, base, rng)
print(json.dumps({"run": "smoke-02", "seconds": round(time.time() - t0, 1), "train_first": losses[0], "train_last": losses[-1], "meaning_same": same, "meaning_drifted": after, "delta": str(delta(table, moved))}))
