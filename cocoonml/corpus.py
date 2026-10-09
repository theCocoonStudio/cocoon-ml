"""The corpus: the complete system's data layer (the completeness list, 2026-10-09).

Reality's data, as the design has it: several languages (tables) in play, each walking; artifacts
produced one after another, each by one table at the state it was in then; a finite number of them,
kept in the order they were produced, with nothing written on an artifact but its forms (the
extension, the table and the state are the generator's and the readers', never the model's). A
window is a stretch of consecutive artifacts, so states succeed within it and tables mix within it;
training samples windows from the corpus in no order (the ball, unordered), and the readers hold the
provenance to read against. There is no clock: the steps a table takes between artifacts and the
switches between tables are drawn, not counted (Poisson counts around a mean).
"""

import math
import random
from dataclasses import dataclass

from cocoonml.schema import delta_magnitude, draw, estimate_shares, step


def _poisson(rng, mean):
    if mean <= 0:
        return 0
    limit, k, product = math.exp(-mean), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k


@dataclass
class Artifact:
    forms: tuple  # what the world produced and kept: the only thing a model sees
    extension: tuple  # the readers' side: the sum of the leaves' weight vectors over the cuts
    table: int  # the readers' side: which language in play produced it
    state: int  # the readers' side: how many steps that language's walk had taken


class Corpus:
    """A finite record of production: `artifacts` in production order, and per table the list of
    states its walk passed through (state 0 the origin), so a reader can place any artifact."""

    def __init__(self, artifacts, walks):
        self.artifacts = artifacts
        self.walks = walks  # walks[table] = [table at state 0, state 1, ...]

    def __len__(self):
        return len(self.artifacts)

    def forms(self):
        return [a.forms for a in self.artifacts]


def produce(tables, rng, bound, lean, radius, count, steps_mean=1.0, switch_mean=0.5):
    """`count` artifacts from the tables in play. Each table walks around its origin within
    `radius` (None: a free walk). Between consecutive artifacts the producing table takes a drawn
    number of steps (Poisson, `steps_mean`) and the production switches to another table with a
    drawn count of switches (Poisson, `switch_mean`; an odd count of switches among two tables is a
    switch); the next table is drawn uniformly among the others. Returns a Corpus."""
    origins = list(tables)
    current = [t for t in origins]
    walks = [[t] for t in origins]
    which = rng.randrange(len(origins))
    artifacts = []
    for _ in range(count):
        if len(origins) > 1 and _poisson(rng, switch_mean) % 2 == 1:
            which = rng.choice([i for i in range(len(origins)) if i != which])
        for _ in range(_poisson(rng, steps_mean)):
            current[which] = step(current[which], rng, bound, lean, centre=origins[which], radius=radius)
            walks[which].append(current[which])
        forms, extension = draw(current[which], rng)
        artifacts.append(Artifact(forms, extension, which, len(walks[which]) - 1))
    return Corpus(artifacts, walks)


def pack_window(artifacts, separator, length):
    """One window from consecutive artifacts: forms joined by the separator, padded with it; targets
    the next token. Only whole artifacts (an artifact that does not fit is left out). Returns
    (tokens, targets, the artifacts in the window). No extension token anywhere: the extension is
    latent, read by the readers from the provenance."""
    tokens, used = [], []
    for a in artifacts:
        if len(tokens) + len(a.forms) + 1 > length:
            break
        tokens.extend(a.forms)
        tokens.append(separator)
        used.append(a)
    while len(tokens) < length:
        tokens.append(separator)
    return tokens, tokens[1:] + [separator], used


def windows(corpus, separator, length, start=0):
    """Every window of the corpus from `start`, consecutive and non-overlapping: the ball as a set of
    windows, each a stretch of production. Training samples these in no order."""
    out, i = [], start
    while i < len(corpus):
        tokens, targets, used = pack_window(corpus.artifacts[i:], separator, length)
        if not used:
            break
        out.append((tokens, targets, used))
        i += len(used)
    return out


def scramble_order(window_artifacts, rng):
    """The control for the latent: the same artifacts in a drawn order, so the succession of states
    and the adjacency of tables are broken while every form and the window's marginal stay. A model
    that reads the index inside the window loses it here; one that reads the marginal does not."""
    shuffled = list(window_artifacts)
    rng.shuffle(shuffled)
    return shuffled


def distance_from_ball(state_table, ball):
    """The truth's distance of a table from a set of tables (the ball training reached): the least
    movement to any of them. A check on the reader's axis, never the axis itself."""
    return min(delta_magnitude(state_table, b) for b in ball)


def estimated_shares(table_shape, forms):
    """The shares a set of strings pins on a shape, as a list per node (None where unvisited)."""
    return [share for share, _, _ in estimate_shares(table_shape, forms)]


def estimated_distance(table_shape, reading_forms, training_forms):
    """The reader's axis: how far the shares a reading window pins sit from the shares the training
    corpus pins, on the same shape, averaged over the nodes both pin; None if they pin none in common.
    Read from artifacts alone, after the fact; the truth (distance_from_ball) is its check."""
    a, b = estimated_shares(table_shape, reading_forms), estimated_shares(table_shape, training_forms)
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(abs(float(x) - float(y)) for x, y in pairs) / len(pairs)
