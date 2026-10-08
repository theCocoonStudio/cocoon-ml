"""The constructed drift source: a table, its draws, its steps, and the delta.

Operational reading of docs/concepts/03-in-context-learning.md (Claude's, for Izzy's strikes):

- A table is a binary grammar. Every nonterminal has two SISTERS, alternative expansions,
  whose shares sum to one (squared moduli of complex amplitudes; a phase per sister).
  Every expansion is a MERGE of two symbols, each a nonterminal or a leaf, with an ORDER
  ratio: the share of draws in which the first symbol is spoken first.
- A leaf is a form on a cut. Form 0 is the empty form (null). Cuts are integers; the
  extension of an artifact is the multiset of cuts it carries, composed as a sum.
- An artifact is one draw of the whole tree, pronounced: the string of non-empty forms.
- A step moves share between sisters by small amounts whose total is bounded, with a
  lean toward the sister that contains a null, and moves phases by small amounts.
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
    return Table(nodes=nodes, resolution=resolution)


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


def step(table, rng, bound, lean):
    """One move of the index: a superposition of small share movements, their total bounded by
    `bound`, leaning by `lean` toward the sister that contains a null; phases move by small
    amounts. Shares that reach zero stay there (remove). Returns a new table."""
    new = table.copy()
    n = len(new.nodes)
    per_node = Fraction(bound, n)
    for node in new.nodes:
        toward_null = [any(_is_nonterminal(s) is False and s.form == 0 for s in (e.first, e.second)) for e in node.sisters]
        direction = Fraction(rng.choice((-1, 1)), 1)
        if toward_null[0] and not toward_null[1]:
            direction = direction if rng.random() >= lean else Fraction(1)
        elif toward_null[1] and not toward_null[0]:
            direction = direction if rng.random() >= lean else Fraction(-1)
        delta = direction * per_node * Fraction(rng.randrange(0, table.resolution + 1), table.resolution)
        node.share = _quantise(min(Fraction(1), max(Fraction(0), node.share + delta)), table.resolution)
        node.phase = (node.phase + Fraction(rng.randrange(-1, 2), 4 * table.resolution)) % 1
        for e in node.sisters:
            e.order = _quantise(min(Fraction(1), max(Fraction(0), e.order + direction * per_node / 2)), table.resolution)
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
# predictive(table, prefix) is the distribution of the next form given a prefix, under interference:
# the squared modulus of the summed amplitudes of derivations consistent with prefix + next, over
# the same for the prefix. T1: a node visited by no derivation consistent with the prefix has no
# effect on this distribution, whatever its share, order or phase. The proof is the definition:
# such a node contributes to no term of either sum. tests/test_schema.py checks it numerically.


def predictive(table, prefix):
    """{next form or None (end): probability} given the prefix, under interference."""
    sums = {}
    for forms, _, choices in _all_derivations(table):
        if forms[: len(prefix)] != tuple(prefix):
            continue
        nxt = forms[len(prefix)] if len(forms) > len(prefix) else None
        sums[nxt] = sums.get(nxt, 0j) + _amplitude(table, choices)
    probs = {k: abs(a) ** 2 for k, a in sums.items()}
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
# Operational reading (Claude, 2026-10-09), marked; Izzy strikes.


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
