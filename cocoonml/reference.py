"""Analytical references for the held loss (decisions 54): the Bayes-optimal in-context predictor
under the generator itself, and the marginal (unigram) predictor, both scored exactly as the model
is (the mean cross-entropy over every position of a window, padding included: array_attention.loss).

The generator is ours (schema.generate, corpus.produce, corpus.pack_window), so the optimum is not a
guess: a window is consecutive artifacts drawn from the table currently producing, the producer
switching between artifacts with an odd Poisson count of switches and stepping a Poisson count of
steps, whole artifacts packed while they fit and the separator padding the rest. With no steps
(radius zero: the stationary case, the setting of docs/literature.md entry 1) the predictor is exact,
a forward pass over the tables in play and a padding hypothesis. With steps the state space is the
walk's (tens of thousands of distinct states in the record) and the predictor is a particle filter
over the same generator: particles carry every table's current state, step with schema.step itself,
and are weighted by the likelihood of each artifact; the estimate converges in the particle count,
a choice marked in the row. The emission of a state is its classical string distribution (the draw
is classical: schema.draw), the extension marginalised."""
import math
import random

from cocoonml.corpus import _poisson
from cocoonml.schema import classical_joint
from cocoonml.schema import step as schema_step

FLOOR = 1e-12  # the smallest probability scored, so a zero never gives an infinite loss


def string_distribution(table):
    """{forms: probability} under the classical draw, the extension marginalised."""
    out = {}
    for (forms, _), p in classical_joint(table).items():
        out[forms] = out.get(forms, 0.0) + p
    return out


def switch_probability(switch_mean, n_tables):
    """P(the producing table changes between consecutive artifacts): an odd Poisson count of
    switches (corpus.produce); among two tables that is the other one, among more the next is
    uniform over the others."""
    if n_tables < 2 or switch_mean <= 0:
        return 0.0
    return math.exp(-switch_mean) * math.sinh(switch_mean)


class SchemaWorld:
    """The schema's tables as a world for the predictor: every mass from the enumerated classical
    string distribution of a state (cached by object identity; states are immutable once produced),
    an artifact fitting when its length plus the separator is within the room left."""

    def __init__(self):
        self._cache = {}

    def _dist(self, table):
        key = id(table)
        if key not in self._cache:
            self._cache[key] = (table, string_distribution(table))  # the table kept so the id stays valid
        return self._cache[key][1]

    def forms(self, origins):
        return sorted({f for t in origins for forms in self._dist(t) for f in forms})

    def _weighted(self, state, weights):
        """The state's string distribution, or its reweighting by a length weight for the window's
        first artifact (decisions 54, second form: the first artifact did not fit the previous window,
        so its length is at least that window's leftover room; `weights` maps a length to the
        probability, over windows, that the leftover admits it)."""
        dist = self._dist(state)
        if weights is None:
            return dist
        key = (id(state), id(weights))
        if key not in self._cache:
            raw = {forms: p * weights(len(forms)) for forms, p in dist.items()}
            total = sum(raw.values())
            self._cache[key] = (state, {forms: p / total for forms, p in raw.items()} if total > 0 else raw)
        return self._cache[key][1]

    def prefix_mass(self, state, prefix, room, weights=None):
        """P(the artifact starts with `prefix` and fits in `room` tokens)."""
        return sum(p for forms, p in self._weighted(state, weights).items() if len(forms) + 1 <= room and forms[: len(prefix)] == prefix)

    def exact_mass(self, state, prefix, room, weights=None):
        """P(the artifact is exactly `prefix` and fits)."""
        return self._weighted(state, weights).get(prefix, 0.0) if len(prefix) + 1 <= room else 0.0

    def empty_mass(self, state, weights=None):
        return self._weighted(state, weights).get((), 0.0)

    def no_fit_mass(self, state, room, weights=None):
        return sum(p for forms, p in self._weighted(state, weights).items() if len(forms) + 1 > room)

    def step(self, state, rng, bound, lean, centre, radius):
        return schema_step(state, rng, bound, lean, centre=centre, radius=radius)


class MarkovWorld:
    """The Markov-chain world (cocoonml/markov.py, decisions 55): a chain's predictive is its row,
    so every mass is a product along the prefix times the probability of ending within the room
    left, by a short recursion over the remaining length; no string is enumerated, and the floor is
    exact with no steps, which is what their KL-to-optimum curve needs."""

    LONGEST = 400  # the length past which a run's mass is treated as zero (every row ends with a grain at least: the tail is geometric)

    def __init__(self):
        self._ends = {}
        self._exact = {}
        self._norm = {}

    def forms(self, origins):
        return list(range(1, origins[0].states + 1))

    def _end_at(self, chain, j, state):
        """P(from `state`, about to transition, exactly j more states are emitted and then the end)."""
        key = (id(chain), j, state)
        if key not in self._exact:
            row = chain.rows[state]
            if j == 0:
                value = float(row[-1])
            else:
                value = sum(float(row[s]) * self._end_at(chain, j - 1, s) for s in range(chain.states) if row[s] > 0)
            self._exact[key] = (chain, value)
        return self._exact[key][1]

    def _norm_of(self, chain, weights):
        """Σ over runs of P(run) · weights(len): the reweighted distribution's mass, so it sums to one."""
        key = (id(chain), id(weights))
        if key not in self._norm:
            total = sum(float(chain.start[s]) * sum(self._end_at(chain, j, s) * weights(1 + j) for j in range(self.LONGEST)) for s in range(chain.states) if chain.start[s] > 0)
            self._norm[key] = (chain, total)
        return self._norm[key][1]

    def _end_within(self, chain, j, state):
        """P(from `state`, about to transition, the run ends with at most j more states emitted)."""
        key = (id(chain), j, state)
        if key not in self._ends:
            row = chain.rows[state]
            end = float(row[-1])
            if j > 0:
                end += sum(float(row[s]) * self._end_within(chain, j - 1, s) for s in range(chain.states) if row[s] > 0)
            self._ends[key] = (chain, end)  # the chain kept so the id stays valid
        return self._ends[key][1]

    def _prefix(self, chain, prefix):
        p = float(chain.start[prefix[0] - 1])
        for a, b in zip(prefix, prefix[1:]):
            p *= float(chain.rows[a - 1][b - 1])
        return p

    def prefix_mass(self, state, prefix, room, weights=None):
        if not prefix:
            return 1.0 - self.no_fit_mass(state, room, weights)
        more = room - 1 - len(prefix)  # states the run may still emit and fit, with its separator
        if more < 0:
            return 0.0
        if weights is None:
            return self._prefix(state, prefix) * self._end_within(state, more, prefix[-1] - 1)
        m = len(prefix)
        return self._prefix(state, prefix) * sum(self._end_at(state, j, prefix[-1] - 1) * weights(m + j) for j in range(more + 1)) / self._norm_of(state, weights)

    def exact_mass(self, state, prefix, room, weights=None):
        if not prefix or len(prefix) + 1 > room:
            return 0.0
        mass = self._prefix(state, prefix) * float(state.rows[prefix[-1] - 1][-1])
        return mass if weights is None else mass * weights(len(prefix)) / self._norm_of(state, weights)

    def empty_mass(self, state, weights=None):
        return 0.0  # a run emits its start state at least

    def no_fit_mass(self, state, room, weights=None):
        if room < 2:
            return 1.0
        if weights is None:
            return 1.0 - sum(float(state.start[s]) * self._end_within(state, room - 2, s) for s in range(state.states) if state.start[s] > 0)
        fits = sum(float(state.start[s]) * sum(self._end_at(state, j, s) * weights(1 + j) for j in range(room - 1)) for s in range(state.states) if state.start[s] > 0)
        return 1.0 - fits / self._norm_of(state, weights)

    def step(self, state, rng, bound, lean, centre, radius):
        from cocoonml.markov import step as markov_step

        return markov_step(state, rng, bound, lean, centre=centre, radius=radius)


def leftover_distribution(window_list, length):
    """The room left after the last whole artifact of each window, as a distribution over windows:
    the window's first artifact did not fit the previous window's leftover (corpus.windows packs
    consecutive artifacts and breaks on the first that does not fit), so its length is at least that
    leftover. The corpus's first window has no previous one and counts as a leftover of zero."""
    counts = {0: 1}
    for tokens, _, used in window_list[:-1]:
        left = length - sum(len(a.forms) + 1 for a in used)
        counts[left] = counts.get(left, 0) + 1
    total = sum(counts.values())
    return {r: c / total for r, c in sorted(counts.items())}


def first_length_weights(leftovers):
    """The weight of a length for the window's first artifact: the probability, over windows, that
    the leftover admitted it (len ≥ leftover); None for no leftovers (an unconditioned first artifact)."""
    if not leftovers:
        return None
    return lambda length: sum(q for r, q in leftovers.items() if length >= r)


class _Hypothesis:
    """One hypothesis about the hidden state: the producing table, every table's current state,
    whether padding has begun, and the weight carried into the current artifact."""

    __slots__ = ("weight", "which", "states", "padded")

    def __init__(self, weight, which, states, padded=False):
        self.weight, self.which, self.states, self.padded = weight, which, states, padded


def window_predictives(tokens, separator, origins, switch_mean, steps_mean=0.0, bound=None, lean=None, radius=None, particles=0, rng=None, world=None, leftovers=None, burn_in=200):
    """Per position i, {next token: probability} under the Bayes-optimal predictor given tokens[:i+1],
    over the forms and the separator. Exact when steps_mean is 0 (hypotheses are the tables in play,
    merged by producer, plus padding); a particle filter otherwise (`particles` of them, `rng` theirs).
    The window's first artifact's producer is uniform over the tables (the switching is symmetric).
    `world` gives the masses (SchemaWorld by default; MarkovWorld for chains). `leftovers`, the
    distribution of the room left before a window's first artifact (leftover_distribution), conditions
    that artifact's length (it did not fit the previous window); None leaves it unconditioned. With
    steps, each particle's tables first take `burn_in` steps of the pulled walk from the origin, so a
    window mid-corpus is predicted from where the walk is, not from where it started (Methuselah,
    2026-10-11, objection 2); the exact path has no steps and no burn-in."""
    world = world or SchemaWorld()
    first_weights = first_length_weights(leftovers)
    n = len(origins)
    p_switch = switch_probability(switch_mean, n)
    exact = steps_mean <= 0
    if not exact and (particles <= 0 or rng is None):
        raise ValueError("with steps the predictor is a particle filter: give `particles` and `rng`")
    if exact:
        hyps = [_Hypothesis(1.0 / n, t, list(origins)) for t in range(n)]
    else:
        # the burn-in: one pulled walk per table, `burn_in` steps from the origin and then a state taken
        # every `thin` steps for each particle, so the particles start spread over the walk's stationary
        # states at a cost of burn_in + thin · particles steps per table (the states are thinned draws
        # of one walk, not independent walks: a choice, marked in row 54)
        thin = 5
        drawn = []
        for t in range(n):
            state = origins[t]
            for _ in range(burn_in):
                state = world.step(state, rng, bound, lean, origins[t], radius)
            taken = []
            for _ in range(particles):
                for _ in range(thin):
                    state = world.step(state, rng, bound, lean, origins[t], radius)
                taken.append(state)
            drawn.append(taken)
        hyps = [_Hypothesis(1.0 / particles, rng.randrange(n), [drawn[t][k] for t in range(n)]) for k in range(particles)]
    forms_seen = world.forms(origins)
    length = len(tokens)
    room = length  # tokens left for the artifact that starts at the current boundary
    prefix = ()
    at_boundary = True
    first = True  # the window's first artifact, whose length the leftovers condition
    out = []

    def transition(hs):
        """Between artifacts: the producer may switch, then steps; exact branching or sampled."""
        new = []
        for h in hs:
            if h.padded or h.weight <= 0:
                new.append(h)
                continue
            if exact:
                new.append(_Hypothesis(h.weight * (1 - p_switch), h.which, h.states))
                for other in range(n):
                    if other != h.which and p_switch > 0:
                        new.append(_Hypothesis(h.weight * p_switch / (n - 1), other, h.states))
            else:
                which = h.which
                if n > 1 and _poisson(rng, switch_mean) % 2 == 1:
                    which = rng.choice([i for i in range(n) if i != which])
                states = list(h.states)
                for _ in range(_poisson(rng, steps_mean)):
                    states[which] = world.step(states[which], rng, bound, lean, origins[which], radius)
                new.append(_Hypothesis(h.weight, which, states))
        if exact:  # merge by (which, padded)
            merged = {}
            for h in new:
                key = (h.which, h.padded)
                if key in merged:
                    merged[key].weight += h.weight
                else:
                    merged[key] = _Hypothesis(h.weight, h.which, h.states, h.padded)
            new = list(merged.values())
        return new

    def normalise(hs):
        total = sum(h.weight for h in hs)
        if total <= 0:
            raise ValueError("the window has probability zero under the generator")
        for h in hs:
            h.weight /= total
        if not exact:
            ess = 1.0 / sum(h.weight ** 2 for h in hs)
            if ess < particles / 2:  # resample when the effective sample size halves (a choice, marked in the row)
                drawn = rng.choices(hs, weights=[h.weight for h in hs], k=particles)
                hs[:] = [_Hypothesis(1.0 / particles, h.which, h.states, h.padded) for h in drawn]
        return hs

    def predictive():
        probs = {}
        weights = first_weights if first else None
        if at_boundary:
            p_sep = 0.0
            for h in hyps:
                if h.padded:
                    p_sep += h.weight
                    continue
                state = h.states[h.which]
                empty = world.empty_mass(state, weights) if room >= 1 else 0.0  # an empty artifact needs the room for its separator; at room 0 it is in no_fit
                p_sep += h.weight * (empty + world.no_fit_mass(state, room, weights))
                for f in forms_seen:
                    probs[f] = probs.get(f, 0.0) + h.weight * world.prefix_mass(state, (f,), room, weights)
            probs[separator] = p_sep
        else:
            # inside an artifact every weight already carries w0 · P(prefix fits): the predictive is the
            # weighted mean of each hypothesis's conditional P(next | prefix, fits)
            total = 0.0
            p_sep = 0.0
            for h in hyps:
                if h.padded or h.weight <= 0:
                    continue
                state = h.states[h.which]
                before = world.prefix_mass(state, prefix, room, weights)
                if before <= 0:
                    continue
                total += h.weight
                p_sep += h.weight * world.exact_mass(state, prefix, room, weights) / before
                for f in forms_seen:
                    probs[f] = probs.get(f, 0.0) + h.weight * world.prefix_mass(state, prefix + (f,), room, weights) / before
            probs[separator] = p_sep
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
        return probs

    for i, tok in enumerate(tokens):
        # observe tokens[i]
        weights = first_weights if first else None
        if at_boundary:
            if tok == separator:
                new = []
                for h in hyps:
                    if h.padded:
                        new.append(h)
                        continue
                    state = h.states[h.which]
                    empty, nofit = (world.empty_mass(state, weights) if room >= 1 else 0.0), world.no_fit_mass(state, room, weights)
                    if exact:  # both branches carried: padding began, or an empty artifact (a boundary again)
                        if nofit > 0:
                            new.append(_Hypothesis(h.weight * nofit, h.which, h.states, padded=True))
                        if empty > 0:
                            new.append(_Hypothesis(h.weight * empty, h.which, h.states))
                    else:  # a particle takes one branch, drawn by their masses, and carries their sum
                        if empty + nofit <= 0:
                            continue
                        padded = rng.random() < nofit / (empty + nofit)
                        new.append(_Hypothesis(h.weight * (empty + nofit), h.which, h.states, padded=padded))
                if not new:
                    raise ValueError("the window has probability zero under the generator")
                hyps = normalise(transition(new) if exact else transition(normalise(new)))
                room -= 1
                first = False  # an empty first artifact (or the padding) ends the first artifact's conditioning
            else:
                for h in hyps:
                    h.weight = 0.0 if h.padded else h.weight * world.prefix_mass(h.states[h.which], (tok,), room, weights)
                if exact:
                    hyps = normalise([h for h in hyps if h.weight > 0])
                prefix = (tok,)
                at_boundary = False
        else:
            if tok == separator:
                for h in hyps:
                    if not h.padded:
                        state = h.states[h.which]
                        before = world.prefix_mass(state, prefix, room, weights)
                        h.weight = h.weight * (world.exact_mass(state, prefix, room, weights) / before if before > 0 else 0.0)
                hyps = normalise(transition(hyps) if exact else transition(normalise(hyps)))
                room -= len(prefix) + 1
                prefix = ()
                at_boundary = True
                first = False
            else:
                for h in hyps:
                    if not h.padded:
                        state = h.states[h.which]
                        before, after = world.prefix_mass(state, prefix, room, weights), world.prefix_mass(state, prefix + (tok,), room, weights)
                        h.weight = h.weight * (after / before if before > 0 else 0.0)
                prefix = prefix + (tok,)
        # predict tokens[i + 1] (the separator after the last position, as pack_window targets it)
        out.append(predictive())
    return out


def bayes_losses(tokens, targets, separator, origins, switch_mean, **kw):
    """-log P(target_i) per position under window_predictives; the window's mean is comparable to the model's loss."""
    return [-math.log(min(1.0, max(FLOOR, probs.get(t, 0.0)))) for probs, t in zip(window_predictives(tokens, separator, origins, switch_mean, **kw), targets)]  # clamped: rounding can put a certain prediction a hair above one


def token_counts(window_list):
    """The unigram counts of the training windows' tokens, padding included: what the marginal predictor learned."""
    counts = {}
    for tokens, _, _ in window_list:
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
    return counts


def marginal_losses(targets, counts):
    """-log P(target_i) per position under the unigram predictor with the given counts."""
    total = sum(counts.values())
    return [-math.log(min(1.0, max(FLOOR, counts.get(t, 0) / total))) for t in targets]
