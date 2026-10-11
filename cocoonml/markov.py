"""The Markov-chain world (decisions 55; docs/literature.md entry 1): Edelman et al.'s data through
our model and readers. A "table" here is a chain: k states as the forms 1..k, a start distribution
over them and a transition matrix whose rows are distributions over the k states and the end
(the separator), every row on the grid of `resolution` grains so a walk can step a grain at a
time as the schema's tables do. An artifact is one run of the chain from its start to its end.
The same interface the corpus layer and the readers use for the schema's tables: draw, step, a
signature, a ball with a distance, and the reader's estimate of a window's distance from forms
alone (the empirical transition rows). Pure; numbers and rngs are the caller's."""
import random
from fractions import Fraction

import numpy as np


class Chain:
    """A chain on the grid: `start` a tuple of k Fractions summing to one; `rows` k tuples of k + 1
    Fractions (the last the end), each summing to one; `resolution` the grid's grains per unit."""

    __slots__ = ("start", "rows", "resolution")

    def __init__(self, start, rows, resolution):
        self.start, self.rows, self.resolution = tuple(start), tuple(tuple(r) for r in rows), resolution

    @property
    def states(self):
        return len(self.start)

    def copy(self):
        return Chain(self.start, self.rows, self.resolution)


def _quantise_row(weights, resolution):
    """A distribution onto the grid: `resolution` grains spread by largest remainder, so the row
    still sums to one and no entry is negative."""
    weights = [max(0.0, float(w)) for w in weights]
    total = sum(weights) or 1.0
    scaled = [w / total * resolution for w in weights]
    floors = [int(x) for x in scaled]
    short = resolution - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for i in order[:short]:
        floors[i] += 1
    return tuple(Fraction(f, resolution) for f in floors)


def generate(states, resolution, seed, alpha=1.0, end_alpha=0.5):
    """A chain drawn from a Dirichlet prior (their generator): the start and each row ~ Dirichlet
    with concentration `alpha` on the states and `end_alpha` on the end, quantised to the grid.
    `end_alpha` below alpha keeps artifacts a few tokens long; at the grid's coarseness an end mass
    of zero is possible, so the end is forced to at least one grain in every row (a choice, marked:
    without it a chain could never end an artifact)."""
    rng = random.Random(seed)
    start = _quantise_row([rng.gammavariate(alpha, 1.0) for _ in range(states)], resolution)
    rows = []
    for _ in range(states):
        row = list(_quantise_row([rng.gammavariate(alpha, 1.0) for _ in range(states)] + [rng.gammavariate(end_alpha, 1.0)], resolution))
        if row[-1] == 0:
            donor = max(range(states), key=lambda i: row[i])
            row[donor] -= Fraction(1, resolution)
            row[-1] = Fraction(1, resolution)
        rows.append(tuple(row))
    return Chain(start, rows, resolution)


def _pick(weights, rng):
    u = rng.random()
    acc = Fraction(0)
    for i, w in enumerate(weights):
        acc += w
        if u < acc:
            return i
    return len(weights) - 1


def draw(chain, rng, start=None):
    """One artifact: the chain run from a drawn start state to its end; forms are the states as
    tokens 1..k (0 is never produced, as in the schema). Returns (forms, extension), the extension
    empty: nothing latent beyond the chain itself."""
    forms = []
    state = _pick(chain.start, rng)
    while True:
        forms.append(state + 1)
        nxt = _pick(chain.rows[state], rng)
        if nxt == chain.states:
            break
        state = nxt
    return tuple(forms), ()


def step(chain, rng, bound, lean=None, centre=None, radius=None):
    """One move of the chain, the schema's step in this world: each row (and the start) moves one
    grain from one entry to another with probability bound / (rows · grain), the receiving entry
    drawn uniformly; with `centre` and `radius`, a row whose distance from the centre's row is d
    moves back toward it with probability min(1, d / radius) instead (the restoring pull the page
    derives). `lean` has no meaning here and is ignored. The end entry never drops below one grain.
    Returns a new Chain."""
    grain = Fraction(1, chain.resolution)
    n_rows = chain.states + 1
    p_move = min(Fraction(1), Fraction(bound) / (n_rows * grain))
    distributions = [list(chain.start)] + [list(r) for r in chain.rows]
    centres = None if centre is None else [list(centre.start)] + [list(r) for r in centre.rows]
    for i, dist in enumerate(distributions):
        if rng.random() >= p_move:
            continue
        end = len(dist) - 1 if i > 0 else None  # the end entry of a transition row keeps at least one grain
        donors = [j for j, w in enumerate(dist) if w >= grain and not (j == end and w <= grain)]
        if not donors:
            continue
        giver = rng.choice(donors)
        receiver = rng.choice([j for j in range(len(dist)) if j != giver])
        if centres is not None and radius:
            displacement = sum(abs(a - b) for a, b in zip(dist, centres[i])) / 2
            if displacement > 0 and rng.random() < min(1, float(displacement / radius)):
                # toward the centre: give from an entry above the centre's to one below it
                above = [j for j, w in enumerate(dist) if w > centres[i][j] and w >= grain and not (j == end and w <= grain)]
                below = [j for j, w in enumerate(dist) if w < centres[i][j]]
                if above and below:
                    giver, receiver = rng.choice(above), rng.choice(below)
        dist[giver] -= grain
        dist[receiver] += grain
    return Chain(distributions[0], distributions[1:], chain.resolution)


def signature(chain):
    return (chain.start, chain.rows)


def distance(a, b):
    """The distance between two chains: the mean over the start and the rows of half the L1
    distance between the distributions (total variation), in [0, 1]."""
    rows_a = [a.start, *a.rows]
    rows_b = [b.start, *b.rows]
    return float(sum(sum(abs(x - y) for x, y in zip(ra, rb)) / 2 for ra, rb in zip(rows_a, rows_b)) / len(rows_a))


class MarkovBall:
    """The states a training walk visited, with the distance of any chain from the nearest, as
    corpus.Ball does for the schema's tables; one array operation per query, cached by signature."""

    def __init__(self, states):
        distinct = {}
        for s in states:
            distinct.setdefault(signature(s), s)
        self.states = list(distinct.values())
        self._cache = {}
        self._rows = np.array([[float(x) for d in (s.start, *s.rows) for x in d] for s in self.states]) if self.states else None
        self._n_rows = (self.states[0].states + 1) if self.states else 1

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)

    def distance(self, chain, phases=True):
        key = signature(chain)
        if key not in self._cache:
            flat = np.array([float(x) for d in (chain.start, *chain.rows) for x in d])
            self._cache[key] = float((np.abs(self._rows - flat).sum(axis=1) / 2 / self._n_rows).min())
        return self._cache[key]


def estimated_rows(states, forms_list):
    """The reader's side: the empirical start distribution and transition rows from strings alone,
    as lists of floats with None for a row never left from (and None for the start if no string)."""
    starts = [0] * states
    counts = [[0] * (states + 1) for _ in range(states)]
    for forms in forms_list:
        if not forms:
            continue
        starts[forms[0] - 1] += 1
        for a, b in zip(forms, forms[1:]):
            counts[a - 1][b - 1] += 1
        counts[forms[-1] - 1][states] += 1
    total = sum(starts)
    start = [s / total for s in starts] if total else None
    rows = [[c / sum(row) for c in row] if sum(row) else None for row in counts]
    return start, rows


def _rows_distance(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(sum(abs(p - q) for p, q in zip(x, y)) / 2 for x, y in pairs) / len(pairs)


def window_distances(chains, window_list, training_forms):
    """corpus.window_distances in this world: per window, the distance of the rows its strings pin
    from the rows the training strings pin (the start included), None where they pin none in common.
    No provenance is used; the chains give only the state count."""
    states = chains[0].states
    t_start, t_rows = estimated_rows(states, training_forms)
    pinned = ([t_start] if t_start is not None else [None]) + t_rows
    out = []
    for _, _, used in window_list:
        w_start, w_rows = estimated_rows(states, [a.forms for a in used])
        out.append(_rows_distance(([w_start] if w_start is not None else [None]) + w_rows, pinned))
    return out
