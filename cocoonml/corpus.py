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
    windows, each a stretch of production. Training samples these in no order. The tail, the stretch
    left when production ends before a window is full, is not a window (found 2026-10-09: it held
    one artifact and set the rank every window reaches to one); it is dropped unless it is the only
    one."""
    out, i = [], start
    shortest = min((len(a.forms) + 1 for a in corpus.artifacts[start:]), default=1)  # the fewest tokens an artifact of this corpus takes
    while i < len(corpus):
        tokens, targets, used = pack_window(corpus.artifacts[i:], separator, length)
        if not used:
            break
        i += len(used)
        room = length - sum(len(a.forms) + 1 for a in used)
        full = i < len(corpus) or room < shortest  # cut by the length, or no artifact of this corpus could have fit
        if full or not out:
            out.append((tokens, targets, used))
    return out


def scramble_order(window_artifacts, rng):
    """The control for the latent: the same artifacts in a drawn order, so the succession of states
    and the adjacency of tables are broken while every form and the window's marginal stay. A model
    that reads the index inside the window loses it here; one that reads the marginal does not."""
    shuffled = list(window_artifacts)
    rng.shuffle(shuffled)
    return shuffled


def _signature(table):
    """A hashable key of a table's values: per node the share, the phase and the orders."""
    return tuple((n.share, n.phase, tuple(e.order for e in n.sisters)) for n in table.nodes)


class Ball:
    """The set of states the training walks visited, held as arrays so a distance is one vectorised
    minimum, and answered once per state. Found 2026-10-09: with the ball as a Python list of the
    42 000 states of a 42 000-artifact corpus, the truth distance per artifact was a Python minimum
    over all of them for every artifact of every reading, and two cells spent hours in the readers
    against twelve minutes of training; the phases jitter every step, so the states rarely repeat
    and deduplication alone did nothing."""

    def __init__(self, states):
        import numpy as np

        distinct = {}
        for state in states:
            distinct.setdefault(_signature(state), state)
        self.states = list(distinct.values())
        self._cache = {}
        if self.states:
            self._ratios = np.array([[float(x) for n in t.nodes for x in (n.share, *(e.order for e in n.sisters))] for t in self.states])
            self._phases = np.array([[float(n.phase) for n in t.nodes] for t in self.states])
            first = self.states[0]
            self._entries_no_phase = sum(1 + len(n.sisters) for n in first.nodes)
            self._entries = self._entries_no_phase + len(first.nodes)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def distance(self, state_table, phases=True):
        """delta_magnitude's least value over the states, exactly, in one array operation."""
        import numpy as np

        key = (_signature(state_table), phases)
        if key not in self._cache:
            ratios = np.array([float(x) for n in state_table.nodes for x in (n.share, *(e.order for e in n.sisters))])
            moved = np.abs(self._ratios - ratios).sum(axis=1)
            if phases:
                d = np.abs(self._phases - np.array([float(n.phase) for n in state_table.nodes]))
                moved = moved + np.minimum(d, 1.0 - d).sum(axis=1)
            self._cache[key] = float(moved.min() / (self._entries if phases else self._entries_no_phase))
        return self._cache[key]


def distance_from_ball(state_table, ball, phases=True):
    """The truth's distance of a table from a set of tables (the ball training reached): the least
    movement to any of them. A check on the reader's axis, never the axis itself. `phases` False
    leaves phase movement out, which no classical artifact carries (schema.delta_magnitude). `ball`
    is a Ball (cached, deduplicated) or any iterable of tables."""
    if isinstance(ball, Ball):
        return ball.distance(state_table, phases)
    return min(delta_magnitude(state_table, b, phases) for b in ball)


def estimated_shares(table_shape, forms):
    """The shares a set of strings pins on a shape, as a list per node (None where unvisited)."""
    return [share for share, _, _ in estimate_shares(table_shape, forms)]


def window_distances(shapes, window_list, training_forms):
    """The reader's axis per window, from artifacts alone: for each window, the estimated distance
    of the shares its strings pin from the shares the training strings pin, averaged over the shapes
    in play that pin something from both; None where none does. No provenance is used: every shape
    reads every string it can parse, in the window and in the corpus (until 2026-10-09 the strings
    were split by the generator's hidden table assignment and pooled over all windows; Methuselah's
    audit, finding 3)."""
    pinned = [estimated_shares(shape, training_forms) for shape in shapes]  # once per shape, not per window
    out = []
    for _, _, used in window_list:
        forms = [a.forms for a in used]
        per_shape = [_shares_distance(estimated_shares(shape, forms), b) for shape, b in zip(shapes, pinned)]
        per_shape = [d for d in per_shape if d is not None]
        out.append(sum(per_shape) / len(per_shape) if per_shape else None)
    return out


def window_truth_distances(window_list, corpus, ball, phases=True):
    """The truth's distance per window: the mean over its artifacts of the distance of the state
    that produced each from the ball. The check on `window_distances`."""
    return [sum(distance_from_ball(corpus.walks[a.table][a.state], ball, phases) for a in used) / len(used) for _, _, used in window_list]


def bin_windows(window_list, keys, bins):
    """The windows split into `bins` groups of equal count by their key (None keys dropped), in
    increasing key order: [(mean key, [windows]) ...]. Readings past the training radius are taken
    on the windows whose states lie past it, not on a mixture (the audit, finding 2): a reading
    corpus walked at a radius holds states from the origin outward."""
    keyed = sorted(((k, w) for k, w in zip(keys, window_list) if k is not None), key=lambda kw: kw[0])
    if not keyed:
        return []
    size = max(1, len(keyed) // bins)
    out = []
    for b in range(bins):
        part = keyed[b * size : (b + 1) * size] if b + 1 < bins else keyed[b * size :]
        if part:
            out.append((sum(k for k, _ in part) / len(part), [w for _, w in part]))
    return out


def estimated_distance(table_shape, reading_forms, training_forms):
    """The reader's axis: how far the shares a reading window pins sit from the shares the training
    corpus pins, on the same shape, averaged over the nodes both pin; None if they pin none in common.
    Read from artifacts alone, after the fact; the truth (distance_from_ball) is its check."""
    return _shares_distance(estimated_shares(table_shape, reading_forms), estimated_shares(table_shape, training_forms))


def _shares_distance(a, b):
    """Mean absolute difference of two share lists over the nodes both pin; None if none."""
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


def form_losses_by_rank_with_error(model, window_list, separator, whole=None):
    """Per rank: (mean form loss, its standard error over the windows, the number of windows with a
    form target at that rank). The error is the band a comparison of two curves is read against (the
    rule table's signs; added 2026-10-09 so a slope or a level has a band, not a bare sign)."""
    per_window = {}
    for tokens, targets, _ in window_list:
        s, c = _by_rank(tokens, targets, position_losses(model, tokens, targets), separator)
        for r in s:
            per_window.setdefault(r, []).append(s[r] / c[r])
    out = []
    for r in sorted(per_window):
        if whole is not None and r >= whole:
            continue
        xs = per_window[r]
        n = len(xs)
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        out.append((mean, math.sqrt(var / n) if n > 1 else None, n))
    return out


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


def probe_rows(model, window_list, corpus, ball, separator, phases=True):
    """Per artifact in the windows: (the attended vector at its separator, the truth's distance of
    its state from `ball`, the artifact's rank in its window, the window's index in `window_list`).
    The x the probe is fitted on and the ys it is asked to read: the distance (the truth, a check on
    whether the activations hold the state) and the rank k (Izzy's "a probe for a variable tracking
    k"); the window index lets a fit split by windows, not rows (the audit, finding 5)."""
    rows = []
    for w, (tokens, _, used) in enumerate(window_list):
        attended = model.probe(tokens)
        seps = [i for i, t in enumerate(tokens) if t == separator]
        for rank, (a, i) in enumerate(zip(used, seps)):
            rows.append((attended[i], distance_from_ball(corpus.walks[a.table][a.state], ball, phases), rank, w))
    return rows


def whole_rank(window_list):
    """The number of whole artifacts every window holds: curves are read only at ranks every window
    reaches (sweep 02 round one)."""
    return min(len(used) for _, _, used in window_list) if window_list else 0
