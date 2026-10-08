# cocoon-ml

Building the LLM paradigm from nothing, one piece per PR. Python standard library only until a GPU is needed.

> **The m.o. is pushing through the information cloud, and every project is a probe into it.**

This repo is one probe. The others (the studio's component library, the records, the physics) feed the same task, and they will move into one monorepo when the time comes. Not yet.

The loop: a concept is written down in `docs/concepts/`, in a few lines. The code for it is written from the concept, not from a reference. The PR is reviewed like any other.

## Run

    python3 -m unittest

No install, no dependencies. Python 3.11 or later.

## Why

`docs/paradigm.md`: what this repo is for, the physics it stands on, the three distinctions, the inversion, and who contributed what.

## Pieces

1. `docs/concepts/01-autograd.md`, `cocoonml/autograd.py`
2. `docs/concepts/02-softmax-cross-entropy.md`, `cocoonml/loss.py`: the loss composed from primitives as the check; the fused loss waits for its derivation
3. `docs/concepts/03-in-context-learning.md` and `docs/reviews/` (the dialectical rounds against the count and the sweeps): the first unsolved problem, its objects, the drift test, both predictions lodged before any run, the count
   - `cocoonml/schema.py`: the constructed drift source: a table, its draws, its steps, the delta, the estimate off artifacts
   - `cocoonml/attention.py`: one layer of causal softmax attention on the scalar autograd, with a probe
   - `cocoonml/harness.py`: the apparatus end to end: contexts packed from a table (form or meaning targets, filling or not), training, the recovery curves by artifact rank, the drifted table
   - `cocoonml/probe.py`: a linear probe in closed form with r-squared, the ruler for reading a variable off the attended vectors
