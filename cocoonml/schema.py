"""The constructed drift source: a table, its draws, its steps, and the delta.

Operational reading of docs/concepts/03-in-context-learning.md (Claude's, for Izzy's strikes):

- A table is a binary grammar. Every nonterminal has two SISTERS, alternative expansions,
  whose shares sum to one (squared moduli of complex amplitudes; a phase per sister).
  Every expansion is a MERGE of two symbols, each a nonterminal or a leaf, with an ORDER
  ratio: the share of draws in which the first symbol is spoken first.
- A leaf is a form on a cut. Form 0 is the empty form (null). Cuts are integers; the
  extension of an artifact is the multiset of cuts it carries, composed as a sum.
- An artifact is one draw of the whole tree, pronounced: the string of non-empty forms.
- A step moves shares and order ratios by one grain at most per node, their expected total
  bounded, with a lean toward the sister that contains a null; the ends reflect, and with a
  centre and a radius the move is pulled back toward the centre (mean-reverting); phases jitter.
- The delta between two tables is a ratio of counts: resolved shares that moved, over
  all resolved shares, per node, at a resolution.

Everything is rational: shares and phases are Fractions; the resolution is a count.
"""

from dataclasses import dataclass, field
from fractions import Fraction
import random


@dataclass(frozen=True)
class Leaf:
    form: int  # 0 is the empty form
    cut: int


@dataclass
class Expansion:
    first: object  # Leaf or nonterminal index (int)
    second: object
    order: Fraction  # share of draws with `first` spoken first


@dataclass
class Node:
    sisters: tuple  # (Expansion, Expansion)
    share: Fraction  # squared modulus of the first sister's amplitude; the second has 1 - share
    phase: Fraction  # relative phase between the sisters, as a fraction of a turn


@dataclass
class Table:
    nodes: list  # Node per nonterminal; nonterminal 0 is the root
    resolution: int  # ρ: distinguishable values of a share or an order ratio
    cuts: int = 2  # how many cuts a leaf can carry (cut indices 0 .. cuts - 1)

    def copy(self):
        return Table(
            nodes=[
                Node(
                    sisters=tuple(Expansion(e.first, e.second, e.order) for e in n.sisters),
                    share=n.share,
                    phase=n.phase,
                )
                for n in self.nodes
            ],
            resolution=self.resolution,
            cuts=self.cuts,
        )


def _is_nonterminal(symbol):
    return isinstance(symbol, int)


def _quantise(x, resolution):
    """A ratio read at the grain: the nearest of `resolution` distinguishable values in [0, 1]."""
    return Fraction(round(x * resolution), resolution)


def generate(forms, cuts, nonterminals, resolution, seed):
    """A table realised from counts: `forms` non-empty forms (plus the empty one), `cuts` cuts,
    `nonterminals` nodes. Nonterminal i expands only into nonterminals > i or leaves, so every
    draw terminates. The last nonterminal expands only into leaves."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2: a ratio needs a value strictly between its ends")
    rng = random.Random(seed)

    def symbol(i):
        deeper = [j for j in range(i + 1, nonterminals)]
        if deeper and rng.random() < Fraction(1, 2):
            return rng.choice(deeper)
        return Leaf(rng.randrange(0, forms + 1), rng.randrange(0, cuts))

    def ratio():
        return Fraction(rng.randrange(1, resolution), resolution)

    nodes = []
    for i in range(nonterminals):
        sisters = tuple(Expansion(symbol(i), symbol(i), ratio()) for _ in range(2))
        nodes.append(Node(sisters=sisters, share=ratio(), phase=Fraction(0)))
    return Table(nodes=nodes, resolution=resolution, cuts=cuts)


def draw(table, rng, start=0):
    """One artifact: the string of non-empty forms, and the extension it carried (the sum of cuts).
    Sisters are drawn by their squared moduli; order by the order ratio. Returns (forms, cuts)."""
    forms, cuts = [], []

    def walk(symbol):
        if not _is_nonterminal(symbol):
            if symbol.form != 0:
                forms.append(symbol.form)
            cuts.append(symbol.cut)
            return
        node = table.nodes[symbol]
        expansion = node.sisters[0] if rng.random() < node.share else node.sisters[1]
        first, second = expansion.first, expansion.second
        if rng.random() >= expansion.order:
            first, second = second, first
        walk(first)
        walk(second)

    walk(start)
    return tuple(forms), tuple(cuts)


def derivations(table, string, start=0, _memo=None):
    """Every derivation of `string` from `start`, each as a tuple of (nonterminal, sister index,
    order index) choices. The count of derivations is the structural ambiguity t of the string;
    the phases of a derivation's choices are what interfere."""
    if _memo is None:
        _memo = {}
    key = (start, string)
    if key in _memo:
        return _memo[key]
    results = []

    def spans(symbol, s):
        """Ways `symbol` can produce exactly `s`: a list of derivation tuples."""
        if not _is_nonterminal(symbol):
            want = () if symbol.form == 0 else (symbol.form,)
            return [()] if s == want else []
        return derivations(table, s, symbol, _memo)

    node = table.nodes[start]
    for si, expansion in enumerate(node.sisters):
        for oi, (a, b) in enumerate(((expansion.first, expansion.second), (expansion.second, expansion.first))):
            for split in range(len(string) + 1):
                left, right = string[:split], string[split:]
                for da in spans(a, left):
                    for db in spans(b, right):
                        results.append(((start, si, oi),) + da + db)
    _memo[key] = results
    return results


def step(table, rng, bound, lean, centre=None, radius=None):
    """One move of the index: a superposition of small moves, one grain per node at most, their
    expected total bounded by `bound` (the bound spread over the nodes, as a probability of moving
    one grain). A share that moves takes one grain in a direction leaned by `lean` toward the sister
    that contains a null, when exactly one of the two does; an order ratio takes one grain in a
    direction of its own. A move past an
    end reflects, so no value is absorbing: a share at zero is a removed sister, and it returns.
    With a `centre` table and a `radius`, a move is pulled toward the centre's value with
    probability displacement over radius, a restoring pull proportional to the displacement: the
    mean-reverting form the page derives, which keeps the walk within about the radius of its
    centre while the lean shifts where it sits. Phases jitter by a quarter grain and wrap.
    Returns a new table."""
    new = table.copy()
    n = len(new.nodes)
    grain = Fraction(1, table.resolution)
    p_move = min(Fraction(1), Fraction(bound) / (n * grain))

    def move(value, direction, centre_value):
        if centre_value is not None and radius:
            displacement = value - centre_value
            if displacement != 0 and rng.random() < min(1, abs(displacement) / radius):
                direction = -1 if displacement > 0 else 1
        moved = value + direction * grain
        if moved > 1:
            moved = 2 - moved
        if moved < 0:
            moved = -moved
        return _quantise(moved, table.resolution)

    for i, node in enumerate(new.nodes):
        toward_null = [any(_is_nonterminal(s) is False and s.form == 0 for s in (e.first, e.second)) for e in node.sisters]
        direction = rng.choice((-1, 1))
        if toward_null[0] and not toward_null[1]:
            direction = direction if rng.random() >= lean else 1
        elif toward_null[1] and not toward_null[0]:
            direction = direction if rng.random() >= lean else -1
        c = centre.nodes[i] if centre is not None else None
        if rng.random() < p_move:
            node.share = move(node.share, direction, c.share if c is not None else None)
        node.phase = (node.phase + Fraction(rng.randrange(-1, 2), 4 * table.resolution)) % 1
        for e, ce in zip(node.sisters, c.sisters if c is not None else (None, None)):
            if rng.random() < p_move:
                e.order = move(e.order, rng.choice((-1, 1)), ce.order if ce is not None else None)
    return new


def delta(a, b):
    """The relative count of resolved ratios that moved between two tables: moved over all,
    counted per node over shares, orders and phases. A ratio of counts."""
    moved = total = 0
    for na, nb in zip(a.nodes, b.nodes):
        total += 1 + len(na.sisters) + 1
        moved += int(na.share != nb.share) + int(na.phase != nb.phase)
        moved += sum(int(ea.order != eb.order) for ea, eb in zip(na.sisters, nb.sisters))
    return Fraction(moved, total)


def estimate(table_shape, artifacts):
    """The incidence and the sister frequencies read off artifacts alone, after the fact:
    which forms occurred (the seen incidence) and, per nonterminal, how often each sister and
    each order was used, from the derivations the strings admit (ambiguous strings count every
    derivation at equal weight). Returns (seen_forms, counts[node] = [[sister0, sister1], [order0, order1]])."""
    seen = set()
    counts = [[[0, 0], [0, 0]] for _ in table_shape.nodes]
    for string in artifacts:
        seen.update(string)
        for d in derivations(table_shape, string):
            for node, si, oi in d:
                counts[node][0][si] += 1
                counts[node][1][oi] += 1
    return seen, counts


# --- Interference: the distribution over strings as squared sums of amplitudes ------------------
# The grammar is acyclic (a nonterminal expands only into higher ones or leaves), so the set of
# derivations is finite and the whole distribution can be enumerated. A derivation's amplitude is
# the product over its choices of the sister's amplitude (√share with its phase) and the order's
# amplitude (√order, real). Strings with several derivations get the squared modulus of the SUM:
# that is where the phases act, and only there.

import cmath
import math


def _amplitude(table, choices):
    amp = complex(1.0, 0.0)
    for node_index, si, oi in choices:
        node = table.nodes[node_index]
        share = float(node.share) if si == 0 else 1.0 - float(node.share)
        phase = 0.0 if si == 0 else 2.0 * math.pi * float(node.phase)
        order = float(node.sisters[si].order)
        order = order if oi == 0 else 1.0 - order
        amp *= math.sqrt(share) * cmath.exp(1j * phase) * math.sqrt(order)
    return amp


def _all_derivations(table, symbol=0):
    """Every (string, cuts, choices) the symbol can produce; finite because the grammar is acyclic."""
    if not _is_nonterminal(symbol):
        forms = () if symbol.form == 0 else (symbol.form,)
        return [(forms, (symbol.cut,), ())]
    node = table.nodes[symbol]
    out = []
    for si, expansion in enumerate(node.sisters):
        left = _all_derivations(table, expansion.first)
        right = _all_derivations(table, expansion.second)
        for oi in (0, 1):
            for fa, ca, da in left:
                for fb, cb, db in right:
                    a, b = ((fa, ca, da), (fb, cb, db)) if oi == 0 else ((fb, cb, db), (fa, ca, da))
                    out.append((a[0] + b[0], a[1] + b[1], ((symbol, si, oi),) + a[2] + b[2]))
    return out


def distribution(table):
    """The probability of each string under interference: |Σ amplitudes over its derivations|²,
    renormalised over the strings. Where no string has two derivations this equals the classical
    draw's distribution; where one does, the readings add as amplitudes, and a silent sister in
    either order is already two readings of one string. The amplitudes are not a unitary evolution,
    so total mass is not conserved under cancellation and the renormalisation is what makes this a
    distribution: a first instance, marked. Returns {string: probability}."""
    sums = {}
    for forms, _, choices in _all_derivations(table):
        sums[forms] = sums.get(forms, 0j) + _amplitude(table, choices)
    probs = {s: abs(a) ** 2 for s, a in sums.items()}
    total = sum(probs.values())
    return {s: p / total for s, p in probs.items()} if total > 0 else probs


def draw_interfering(table, rng, _cache=None):
    """One artifact drawn from the interfering distribution; the cuts are not returned, since under
    interference a string is not one derivation and carries no single extension."""
    dist = distribution(table) if _cache is None else _cache
    r = rng.random()
    acc = 0.0
    for s, p in dist.items():
        acc += p
        if r < acc:
            return s
    return s


# --- The predictive distribution, and the first theorem of the toy ------------------------------
# predictive(table, prefix) is the distribution of the next form given a prefix: the conditional of
# `distribution`. Interference acts among the derivations of ONE string; strings that share a prefix
# are distinguishable outcomes and add as probabilities. (As first written, 2026-10-08 morning,
# predictive summed amplitudes across every consistent derivation of every string with that prefix
# and squared the sum, so two continuations of one prefix interfered; found at the fresh session's
# read the same day and corrected; the count review, round one, item 9.) T1: a node visited by no
# derivation consistent with the prefix has no effect on this distribution, whatever its share,
# order or phase: it enters no amplitude of any consistent string, and the normaliser over all
# strings cancels in the conditional. tests/test_schema.py checks it numerically.


def predictive(table, prefix):
    """{next form or None (end): probability} given the prefix, the conditional of distribution."""
    probs = {}
    for forms, p in distribution(table).items():
        if p <= 0 or forms[: len(prefix)] != tuple(prefix):
            continue  # a string of probability zero is not one the world produces
        nxt = forms[len(prefix)] if len(forms) > len(prefix) else None
        probs[nxt] = probs.get(nxt, 0.0) + p
    total = sum(probs.values())
    return {k: p / total for k, p in probs.items()} if total > 0 else probs


def visited_by_consistent_derivations(table, prefix):
    """The set of nodes visited by at least one derivation consistent with the prefix."""
    nodes = set()
    for forms, _, choices in _all_derivations(table):
        if forms[: len(prefix)] == tuple(prefix):
            nodes.update(node for node, _, _ in choices)
    return nodes


# --- The three projections of one carrier ------------------------------------------------------
# The table's values are complex amplitudes (Gaussian rationals in principle; floats here). A
# reader takes projections: FREQUENCY is the squared modulus; COST is minus the log of the modulus,
# which under coarse resolution turns (plus, times) into (min, plus), the tropical semiring; PHASE
# is what both of those discard. None of these is a separate table; they are the same amplitudes
# read three ways. The extension under the carrier: each derivation's cuts weighted by its
# amplitude, so meaning rides on the same values as form.


def projections(amplitude):
    """(frequency, cost, phase) of one amplitude."""
    r = abs(amplitude)
    return r * r, (-math.log(r) if r > 0 else math.inf), cmath.phase(amplitude)


def weighted_extension(table, string):
    """The extension of a string under interference: for each cut, the squared modulus of the
    summed amplitudes of the derivations of `string` that carry it, normalised over cuts present.
    With one derivation this is the plain multiset of its cuts, each at weight one."""
    sums = {}
    for forms, cuts, choices in _all_derivations(table):
        if forms != tuple(string):
            continue
        a = _amplitude(table, choices)
        for c in set(cuts):
            sums[c] = sums.get(c, 0j) + a * cuts.count(c)
    weights = {c: abs(a) ** 2 for c, a in sums.items()}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()} if total > 0 else weights


def cost_of(table, string):
    """The cost projection of a string: minus the log of the modulus of its summed amplitude."""
    total = 0j
    for forms, _, choices in _all_derivations(table):
        if forms == tuple(string):
            total += _amplitude(table, choices)
    r = abs(total)
    return -math.log(r) if r > 0 else math.inf


# --- n_eff: what a window pins ----------------------------------------------------------------
# The independent projections of the carrier are two: the modulus and the phase (cost is the
# modulus read at coarse grain, pinned whenever the modulus is). For a window of artifacts:
#   modulus grade at a node = min(1, √visits / ρ)        (a frequency is pinned to about √n levels)
#   phase grade at a node   = min(1, √ambiguous_visits / φ) (a phase shows only through strings
#                                                            with more than one derivation)
# n_eff for the window is the sum of grades over nodes, over 2 · nodes: a ratio of counts.
# Operational reading (Claude, 2026-10-08), marked; Izzy strikes.


def n_eff(table, artifacts, phase_resolution):
    """(n_eff as a Fraction-like float in [0, 1], per-node grades [(modulus, phase), ...])."""
    visits = [0] * len(table.nodes)
    ambiguous = [0] * len(table.nodes)
    for string in artifacts:
        # a derivation of zero amplitude (a share or an order at zero) is not a reading the world
        # produces, so ambiguity is counted over the readings with nonzero amplitude
        ds = [d for d in derivations(table, string) if abs(_amplitude(table, d)) > 0]
        for d in ds:
            for node, _, _ in d:
                visits[node] += 1
                if len(ds) > 1:
                    ambiguous[node] += 1
    grades = []
    for node_index in range(len(table.nodes)):
        m = min(1.0, math.sqrt(visits[node_index]) / table.resolution)
        p = min(1.0, math.sqrt(ambiguous[node_index]) / phase_resolution) if phase_resolution > 0 else 0.0
        grades.append((m, p))
    total = sum(m + p for m, p in grades)
    return total / (2 * len(table.nodes)), grades


# --- T2: the weighted T1 ---------------------------------------------------------------------
# How much a node visited by SOME consistent derivations moves the prediction. Let m be the share
# of the consistent squared-amplitude mass that passes through the node (its consistent mass),
# s its share, δ a move of that share. Claim (sketch): the total variation of the predictive
# distribution is at most m · |δ| / min(s, 1 − s) to first order in δ. T1 is the case m = 0.
# consistent_mass and sensitivity below let a test check the claim on random tables.


def consistent_mass(table, prefix, node_index):
    """The share of consistent squared-amplitude mass whose derivations visit the node."""
    through = total = 0.0
    for forms, _, choices in _all_derivations(table):
        if forms[: len(prefix)] != tuple(prefix):
            continue
        w = abs(_amplitude(table, choices)) ** 2
        total += w
        if any(n == node_index for n, _, _ in choices):
            through += w
    return through / total if total > 0 else 0.0


def sensitivity(table, prefix, node_index, delta):
    """Total variation distance between the predictive distributions before and after moving the
    node's share by delta (clipped to [0, 1])."""
    before = predictive(table, prefix)
    moved = table.copy()
    node = moved.nodes[node_index]
    node.share = min(Fraction(1), max(Fraction(0), node.share + delta))
    after = predictive(moved, prefix)
    keys = set(before) | set(after)
    return 0.5 * sum(abs(before.get(k, 0.0) - after.get(k, 0.0)) for k in keys)


# --- The delta as a magnitude, not a flag ------------------------------------------------------
# delta() counts whether a resolved ratio moved, which saturates after one step: every node moves
# a little, so nearly every entry flags. delta_magnitude is the mean absolute move of shares,
# orders and phases, in units of the grain: a ratio of counts that keeps growing with drift.


def delta_magnitude(a, b):
    """Mean absolute movement of shares, orders and phases between two tables, as a float."""
    moved = total = 0.0
    for na, nb in zip(a.nodes, b.nodes):
        moved += abs(float(na.share) - float(nb.share))
        d = abs(float(na.phase) - float(nb.phase))
        moved += min(d, 1.0 - d)
        moved += sum(abs(float(ea.order) - float(eb.order)) for ea, eb in zip(na.sisters, nb.sisters))
        total += 2 + len(na.sisters)
    return moved / total if total else 0.0


# --- Identification error: what a window pins, as a distance -----------------------------------
# n_eff counts visits, so two tables with the same readings and different shares read the same to
# it (sweep 02, round one, item 2). The identification error is the distance between the shares a
# reader estimates off the window's artifacts and the table's own, per node: what n_eff was taken
# for. The reader has the shape and the strings, nothing else: every derivation the shape admits
# counts, the derivations of one string at equal weight summing to one, so a string is one
# observation. Orders are estimated per sister, where the reading is defined; a silent daughter
# leaves the order unidentifiable, which the estimate shows as one half.


def estimate_shares(table_shape, artifacts):
    """Per node: (share estimate, its weight, [(order estimate, weight) per sister]); an estimate
    is None where no derivation visited it. Weights are observations, strings counting once. A
    string the shape cannot produce has no derivation and counts as nothing."""
    n = len(table_shape.nodes)
    sister_w = [[0.0, 0.0] for _ in range(n)]
    order_w = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(n)]
    for string in artifacts:
        ds = derivations(table_shape, string)
        if not ds:
            continue
        w = 1.0 / len(ds)
        for d in ds:
            for node, si, oi in d:
                sister_w[node][si] += w
                order_w[node][si][oi] += w
    out = []
    for i in range(n):
        total = sister_w[i][0] + sister_w[i][1]
        share = (sister_w[i][0] / total, total) if total > 0 else (None, 0.0)
        orders = []
        for si in range(2):
            t = order_w[i][si][0] + order_w[i][si][1]
            orders.append((order_w[i][si][0] / t, t) if t > 0 else (None, 0.0))
        out.append((share[0], share[1], orders))
    return out


def identification_error(table, artifacts):
    """(mean absolute share error over the nodes the window identifies, or None; coverage, the
    identified fraction of nodes; per node (share error or None, [order error or None per sister])).
    The error is the reader's whole error: sampling error, and the ambiguity error, since a string
    that does not tell the sisters apart at a node pulls the estimate toward one half however many
    such strings the window holds (the empty world: every string empty, the root read at one half)."""
    estimates = estimate_shares(table, artifacts)
    per_node, errors = [], []
    for node, (share, _, orders) in zip(table.nodes, estimates):
        e = abs(share - float(node.share)) if share is not None else None
        oe = [abs(o - float(sister.order)) if o is not None else None for (o, _), sister in zip(orders, node.sisters)]
        per_node.append((e, oe))
        if e is not None:
            errors.append(e)
    mean = sum(errors) / len(errors) if errors else None
    return mean, len(errors) / len(table.nodes), per_node


# --- The exact entropy floors -------------------------------------------------------------------
# The harness samples the table classically (draw), so the irreducible loss of its targets is a
# conditional entropy under the classical joint over (string, extension), enumerable because the
# grammar is acyclic. Two floors for the two readouts: FORM, the entropy of the next form given the
# forms before it in the artifact, per form target as form_loss_by_rank counts them (the first form
# and the end are not form targets); MEANING, the entropy of the extension given the whole string,
# one target per artifact. A loss minus its floor is the excess, comparable across tables of
# different entropy (sweep 02 round one, item 5); the irreducible term of the decomposition in
# 2401.15530, exact here rather than estimated.


def classical_joint(table):
    """{(string, extension): probability} under the classical draw: shares and orders as
    probabilities, phases absent."""
    joint = {}
    for forms, cuts, choices in _all_derivations(table):
        p = 1.0
        for node_index, si, oi in choices:
            node = table.nodes[node_index]
            p *= float(node.share) if si == 0 else 1.0 - float(node.share)
            o = float(node.sisters[si].order)
            p *= o if oi == 0 else 1.0 - o
        key = (forms, sum(cuts))
        joint[key] = joint.get(key, 0.0) + p
    return joint


def entropy_floor_form(table):
    """Expected -log P(next form | the forms before it), per form target, positions two to the
    end of the string; the floor of form_loss_by_rank. Zero when the first form fixes the string."""
    strings = {}
    for (forms, _), p in classical_joint(table).items():
        if p > 0:  # a derivation through a zero share or order is not a string the world produces
            strings[forms] = strings.get(forms, 0.0) + p
    prefix = {}
    for forms, p in strings.items():
        for i in range(len(forms) + 1):
            prefix[forms[:i]] = prefix.get(forms[:i], 0.0) + p
    total = targets = 0.0
    for forms, p in strings.items():
        for i in range(1, len(forms)):
            total += p * -math.log(prefix[forms[: i + 1]] / prefix[forms[:i]])
            targets += p
    return total / targets if targets > 0 else 0.0


def entropy_floor_meaning(table):
    """Expected entropy of the extension given the whole string, one target per artifact; the
    floor of meaning_loss_at_separators. Zero when every string carries one extension."""
    joint = {k: p for k, p in classical_joint(table).items() if p > 0}
    strings = {}
    for (forms, _), p in joint.items():
        strings[forms] = strings.get(forms, 0.0) + p
    return sum(p * -math.log(p / strings[forms]) for (forms, _), p in joint.items())


# --- A move on the incidence (sweep 03 round one, item 3) ---------------------------------------
# step moves shares, orders and phases: which strings occur. The form–cut pairs, the incidence, it
# never touches, so no run produces a pair the training table did not hold, which is where the two
# predictions separate (I_Δ in the count). remap is that move, kept apart from step so that the two
# axes of the page's test, form drift and protocol drift, are two knobs. It gives `count` spoken
# leaves another cut the table already uses and leaves every share and order as it was, so the
# string distribution is unchanged and only the extensions move.


def cuts_of(table):
    """The cuts the table's leaves use, sorted."""
    return sorted({leaf.cut for node in table.nodes for e in node.sisters for leaf in (e.first, e.second) if not _is_nonterminal(leaf)})


def remap(table, rng, count):
    """A new table with `count` distinct spoken leaves (form not empty) given another of the
    table's cuts. Returns (table, moves), a move being (node, sister, slot, old cut, new cut).
    Refuses a table with fewer than two cuts or fewer spoken leaves than `count`."""
    cuts = cuts_of(table)
    if len(cuts) < 2:
        raise ValueError("remap needs a table with at least two cuts")
    new = table.copy()
    spoken = [
        (i, si, slot)
        for i, node in enumerate(new.nodes)
        for si, e in enumerate(node.sisters)
        for slot, leaf in enumerate((e.first, e.second))
        if not _is_nonterminal(leaf) and leaf.form != 0
    ]
    if len(spoken) < count:
        raise ValueError(f"remap of {count} leaves on a table with {len(spoken)} spoken leaves")
    moves = []
    for i, si, slot in rng.sample(spoken, count):
        e = new.nodes[i].sisters[si]
        leaf = e.first if slot == 0 else e.second
        new_cut = rng.choice([c for c in cuts if c != leaf.cut])
        moved = Leaf(leaf.form, new_cut)
        if slot == 0:
            e.first = moved
        else:
            e.second = moved
        moves.append((i, si, slot, leaf.cut, new_cut))
    return new, moves


def incidence_delta(a, b):
    """The fraction of leaf slots whose cut differs between two tables of one shape: a ratio of
    counts, the I_Δ of the count read off the tables."""
    moved = total = 0
    for na, nb in zip(a.nodes, b.nodes):
        for ea, eb in zip(na.sisters, nb.sisters):
            for la, lb in ((ea.first, eb.first), (ea.second, eb.second)):
                if _is_nonterminal(la):
                    continue
                total += 1
                moved += int(la.cut != lb.cut)
    return Fraction(moved, total) if total else Fraction(0)


# --- The identification excess (sweep 04 round one, item 4) --------------------------------------
# A share at an end of its range is identified from a handful of draws with no error, so the
# identification error is lower on low-entropy tables for no reason of the window's. The excess is
# the error less the sampling floor at that node: the expected absolute error of a frequency at the
# true share and the visit count, √(2/π) · √(s(1 − s)/n) (the mean absolute deviation of a normal
# with the frequency's standard error; sweep 05's first cells read the excess slightly below zero
# with the standard error as the floor, which is this factor). Zero excess means the window pinned
# the share as well as n draws can; the excess, not the error, compares across tables.


def identification_excess(table, artifacts):
    """(mean of (error − floor) over the identified nodes, or None; per node (excess or None))."""
    estimates = estimate_shares(table, artifacts)
    per_node, values = [], []
    for node, (share, weight, _) in zip(table.nodes, estimates):
        if share is None:
            per_node.append(None)
            continue
        s = float(node.share)
        floor = math.sqrt(2.0 / math.pi) * math.sqrt(s * (1.0 - s) / weight) if weight > 0 else 0.0
        excess = abs(share - s) - floor
        per_node.append(excess)
        values.append(excess)
    return (sum(values) / len(values) if values else None), per_node


def max_extension_size(table):
    """The most cuts any derivation of the table carries: the largest extension, which bounds the
    extension vocabulary (`harness.extension_vocabulary`)."""
    return max(len(cuts) for _, cuts, _ in _all_derivations(table))
