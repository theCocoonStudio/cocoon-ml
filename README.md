# cocoon-ml

Building the LLM paradigm from nothing, one piece per PR. Python standard library only until a GPU is needed.

> **The m.o. is pushing through the information cloud, and every project is a probe into it.**

This repo is one probe. The others (the studio's component library, the records, the physics) feed the same task, and they will move into one monorepo when the time comes. Not yet.

The loop: a concept is written down in `docs/concepts/`, in a few lines. The code for it is written from the concept, not from a reference. The PR is reviewed like any other.

## Run

    python3 -m unittest

No install, no dependencies. Python 3.11 or later.

## Pieces

1. `docs/concepts/01-autograd.md`, `cocoonml/autograd.py`
