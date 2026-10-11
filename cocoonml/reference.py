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

    def prefix_mass(self, state, prefix, room):
        """P(the artifact starts with `prefix` and fits in `room` tokens)."""
        return sum(p for forms, p in self._dist(state).items() if len(forms) + 1 <= room and forms[: len(prefix)] == prefix)

    def exact_mass(self, state, prefix, room):
        """P(the artifact is exactly `prefix` and fits)."""
        return self._dist(state).get(prefix, 0.0) if len(prefix) + 1 <= room else 0.0

    def empty_mass(self, state):
        return self._dist(state).get((), 0.0)

    def no_fit_mass(self, state, room):
        return sum(p for forms, p in self._dist(state).items() if len(forms) + 1 > room)

    def step(self, state, rng, bound, lean, centre, radius):
        return schema_step(state, rng, bound, lean, centre=centre, radius=radius)


class MarkovWorld:
    """The Markov-chain world (cocoonml/markov.py, decisions 55): a chain's predictive is its row,
    so every mass is a product along the prefix times the probability of ending within the room
    left, by a short recursion over the remaining length; no string is enumerated, and the floor is
    exact with no steps, which is what their KL-to-optimum curve needs."""

    def __init__(self):
        self._ends = {}

    def forms(self, origins):
        return list(range(1, origins[0].states + 1))

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

    def prefix_mass(self, state, prefix, room):
        if not prefix:
            return 1.0 - self.no_fit_mass(state, room)
        more = room - 1 - len(prefix)  # states the run may still emit and fit, with its separator
        if more < 0:
            return 0.0
        return self._prefix(state, prefix) * self._end_within(state, more, prefix[-1] - 1)

    def exact_mass(self, state, prefix, room):
        if not prefix or len(prefix) + 1 > room:
            return 0.0
        return self._prefix(state, prefix) * float(state.rows[prefix[-1] - 1][-1])

    def empty_mass(self, state):
        return 0.0  # a run emits its start state at least

    def no_fit_mass(self, state, room):
        if room < 2:
            return 1.0
        return 1.0 - sum(float(state.start[s]) * self._end_within(state, room - 2, s) for s in range(state.states) if state.start[s] > 0)

    def step(self, state, rng, bound, lean, centre, radius):
        from cocoonml.markov import step as markov_step

        return markov_step(state, rng, bound, lean, centre=centre, radius=radius)


class _Hypothesis:
    """One hypothesis about the hidden state: the producing table, every table's current state,
    whether padding has begun, and the weight carried into the current artifact."""

    __slots__ = ("weight", "which", "states", "padded")

    def __init__(self, weight, which, states, padded=False):
        self.weight, self.which, self.states, self.padded = weight, which, states, padded


def window_predictives(tokens, separator, origins, switch_mean, steps_mean=0.0, bound=None, lean=None, radius=None, particles=0, rng=None, world=None):
    """Per position i, {next token: probability} under the Bayes-optimal predictor given tokens[:i+1],
    over the forms and the separator. Exact when steps_mean is 0 (hypotheses are the tables in play,
    merged by producer, plus padding); a particle filter otherwise (`particles` of them, `rng` theirs).
    The window's first artifact's producer is uniform over the tables (the switching is symmetric).
    `world` gives the masses (SchemaWorld by default; MarkovWorld for chains)."""
    world = world or SchemaWorld()
    n = len(origins)
    p_switch = switch_probability(switch_mean, n)
    exact = steps_mean <= 0
    if not exact and (particles <= 0 or rng is None):
        raise ValueError("with steps the predictor is a particle filter: give `particles` and `rng`")
    if exact:
        hyps = [_Hypothesis(1.0 / n, t, list(origins)) for t in range(n)]
    else:
        hyps = [_Hypothesis(1.0 / particles, rng.randrange(n), list(origins)) for _ in range(particles)]
    forms_seen = world.forms(origins)
    length = len(tokens)
    room = length  # tokens left for the artifact that starts at the current boundary
    prefix = ()
    at_boundary = True
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
        if at_boundary:
            p_sep = 0.0
            for h in hyps:
                if h.padded:
                    p_sep += h.weight
                    continue
                state = h.states[h.which]
                p_sep += h.weight * (world.empty_mass(state) + world.no_fit_mass(state, room))
                for f in forms_seen:
                    probs[f] = probs.get(f, 0.0) + h.weight * world.prefix_mass(state, (f,), room)
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
                before = world.prefix_mass(state, prefix, room)
                if before <= 0:
                    continue
                total += h.weight
                p_sep += h.weight * world.exact_mass(state, prefix, room) / before
                for f in forms_seen:
                    probs[f] = probs.get(f, 0.0) + h.weight * world.prefix_mass(state, prefix + (f,), room) / before
            probs[separator] = p_sep
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
        return probs

    for i, tok in enumerate(tokens):
        # observe tokens[i]
        if at_boundary:
            if tok == separator:
                new = []
                for h in hyps:
                    if h.padded:
                        new.append(h)
                        continue
                    state = h.states[h.which]
                    empty, nofit = world.empty_mass(state), world.no_fit_mass(state, room)
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
            else:
                for h in hyps:
                    h.weight = 0.0 if h.padded else h.weight * world.prefix_mass(h.states[h.which], (tok,), room)
                if exact:
                    hyps = normalise([h for h in hyps if h.weight > 0])
                prefix = (tok,)
                at_boundary = False
        else:
            if tok == separator:
                for h in hyps:
                    if not h.padded:
                        state = h.states[h.which]
                        before = world.prefix_mass(state, prefix, room)
                        h.weight = h.weight * (world.exact_mass(state, prefix, room) / before if before > 0 else 0.0)
                hyps = normalise(transition(hyps) if exact else transition(normalise(hyps)))
                room -= len(prefix) + 1
                prefix = ()
                at_boundary = True
            else:
                for h in hyps:
                    if not h.padded:
                        state = h.states[h.which]
                        before, after = world.prefix_mass(state, prefix, room), world.prefix_mass(state, prefix + (tok,), room)
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
