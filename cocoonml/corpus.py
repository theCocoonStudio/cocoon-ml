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


# --- Readers over corpus windows (the latent read on form) --------------------------------------
# The extension never enters the input, so meaning is read where a language model's is: on the
# forms. Three readers: the form loss by artifact rank within a window; the order control (the same
# artifacts in a drawn order: what the model loses when the succession of states is broken is what
# it was reading of the index); and the probe's rows (the attended vector at each artifact's
# separator against the truth's distance of that artifact's state from the training ball).


def position_losses(model, tokens, targets):
    """-log p(target) at every position, as floats. A model with its own `position_losses` (the
    array model) answers directly; the scalar model's logits are read as before."""
    if hasattr(model, "position_losses"):
        return list(model.position_losses(tokens, targets))
    logits, _ = model.forward(tokens)
    out = []
    for lg, t in zip(logits, targets):
        m = max(x.value for x in lg)
        total = sum(math.exp(x.value - m) for x in lg)
        out.append(-(lg[t].value - m) + math.log(total))
    return out


def _by_rank(tokens, targets, losses, separator):
    """Form losses grouped by artifact rank: positions that are separators, or whose target is a
    separator (the end of an artifact), are not form predictions."""
    sums, counts, rank = {}, {}, 0
    for tok, t, loss in zip(tokens, targets, losses):
        if tok == separator:
            rank += 1
            continue
        if t == separator:
            continue
        sums[rank] = sums.get(rank, 0.0) + loss
        counts[rank] = counts.get(rank, 0) + 1
    return sums, counts


def form_losses_by_rank(model, window_list, separator, whole=None):
    """Mean form loss by artifact rank over the windows [(tokens, targets, used)]; ranks past
    `whole` (the rank every window reaches) are dropped when given."""
    sums, counts = {}, {}
    for tokens, targets, _ in window_list:
        s, c = _by_rank(tokens, targets, position_losses(model, tokens, targets), separator)
        for r in s:
            sums[r] = sums.get(r, 0.0) + s[r]
            counts[r] = counts.get(r, 0) + c[r]
    return [sums[r] / counts[r] for r in sorted(sums) if whole is None or r < whole]


def order_control(model, window_list, separator, rng, whole=None):
    """By rank: the form loss of the same artifacts in a drawn order minus the loss in the order
    produced. Positive where the model read the succession of states; zero where it read only the
    window's marginal."""
    sums_o, counts_o, sums_s = {}, {}, {}
    for tokens, targets, used in window_list:
        s, c = _by_rank(tokens, targets, position_losses(model, tokens, targets), separator)
        t2, g2, _ = pack_window(scramble_order(used, rng), separator, len(tokens))
        s2, _ = _by_rank(t2, g2, position_losses(model, t2, g2), separator)
        for r in s:
            sums_o[r] = sums_o.get(r, 0.0) + s[r]
            counts_o[r] = counts_o.get(r, 0) + c[r]
            sums_s[r] = sums_s.get(r, 0.0) + s2.get(r, 0.0)
    return [(sums_s[r] - sums_o[r]) / counts_o[r] for r in sorted(sums_o) if whole is None or r < whole]


def probe_rows(model, window_list, corpus, ball, separator):
    """Per artifact in the windows: (the attended vector at its separator, the truth's distance of
    its state from `ball`). The x the probe is fitted on and the y it is asked to read."""
    rows = []
    for tokens, _, used in window_list:
        attended = model.probe(tokens)
        seps = [i for i, t in enumerate(tokens) if t == separator]
        for a, i in zip(used, seps):
            rows.append((attended[i], distance_from_ball(corpus.walks[a.table][a.state], ball)))
    return rows


def whole_rank(window_list):
    """The number of whole artifacts every window holds: curves are read only at ranks every window
    reaches (sweep 02 round one)."""
    return min(len(used) for _, _, used in window_list) if window_list else 0
