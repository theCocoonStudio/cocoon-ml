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
