# cocoon-ml

Building the LLM paradigm from nothing, one piece per PR. Python standard library, plus numpy for the array model since 2026-10-09 (Debian's `python3-numpy`; the scalar model stays the oracle, the array model is the apparatus that runs in minutes), until a GPU is needed.

> **The m.o. is pushing through the information cloud, and every project is a probe into it.**

This repo is one probe. The others (the studio's component library, the records, the physics) feed the same task, and they will move into one monorepo when the time comes. Not yet.

The loop: a concept is written down in `docs/concepts/`, in a few lines. The code for it is written from the concept, not from a reference. The PR is reviewed like any other.

## Run

    python3 -m unittest

No install; numpy from the distribution (`python3-numpy`; CI installs the same version from PyPI, pinned in `.github/workflows/check.yml`). Nothing unlinted: `python3 -m pyflakes cocoonml tests runs` passes before a commit (Debian's `python3-pyflakes`, 2.5.0, the version CI pins), then `python3 -m unittest`. Python 3.11 or later. Since 2026-10-19 the suite also refuses a tracked record of a session or a chat and a project `.npmrc` (`tests/test_structure.py`; the `records-guard` workflow refuses the same at the push), and `CODEOWNERS` makes the workflow files the owner's to review; `.github/workflows/reviewer-policy.yml` is the reviewer policy every other repository carries (who may approve whose pull request, the job `policy` the verdict), required in the ruleset once the owner switches it on. The suite runs a toy-size cell of `runs/sweep7.py` in both modes as part of its tests.

## Why

`docs/paradigm.md`: what this repo is for, the physics it stands on, the three distinctions, the inversion, and who contributed what.

## Pieces

1. `docs/concepts/01-autograd.md`, `cocoonml/autograd.py`
2. `docs/concepts/02-softmax-cross-entropy.md`, `cocoonml/loss.py`: the loss composed from primitives as the check; the fused loss waits for its derivation
3. `docs/concepts/03-in-context-learning.md` and `docs/reviews/` (the dialectical rounds against the count, the sweeps, the factors and the run's row): the first unsolved problem, its objects, the drift test, both predictions lodged before any run, the count. `docs/decisions.md`: every implementation decision, one row each with its forms after the rounds, for oversight; since 2026-10-10 a row can be marked Claude's to make (53). `docs/rules.md`: the rule table the runner moves the parameter vector by. `docs/literature.md`: the bridge to the field, one entry per paper with what it predicts for our data and our replication. `runs/`: how every sweep was launched, with its launch scripts, and `runs/row53.py`, the reading of the run of decisions 53.
   - `cocoonml/schema.py`: the constructed drift source: a table, its draws, its steps, the delta, the estimate off artifacts
   - `cocoonml/attention.py`: one layer of causal softmax attention on the scalar autograd, with a probe
   - `cocoonml/harness.py`: the apparatus end to end: contexts packed from a table (form or meaning targets, filling or not), training, the recovery curves by artifact rank, the drifted table
   - `cocoonml/probe.py`: a linear probe in closed form with r-squared, the ruler for reading a variable off the attended vectors
   - `cocoonml/corpus.py`: the corpus layer: graded tables walked within a training radius, a finite corpus produced in order, windows, the readers' bins by distance
   - `cocoonml/array_attention.py`: the scalar model on numpy arrays with the backward in closed form, checked against the scalar; the derived initialisation (decisions 41); its attention weights exposed for the circuits reader
   - `cocoonml/rules.py`: the rule table as pure functions, and decisions 53's predicates and tests (the onset by the running median and the null's envelope, the second descent, the ladder's concordance, the rank sign, the pairwise differ, the crossing and the cap)
   - `cocoonml/run_row.py`: decisions 53 applied to a record, each outcome mapped to its decision
   - `cocoonml/circuits.py`: the previous-token and induction scores on a copy's attention (`docs/literature.md`, entries 2 and 3)
