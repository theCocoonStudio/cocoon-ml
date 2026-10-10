import math
from fractions import Fraction
"""The apparatus end to end: artifacts from a table, packed into one context, a model trained
on contexts from the training table and read on contexts from a drifted table, position by
position. A later position has seen more artifacts, so loss by position is the recovery curve
against k, read inside one window. Nothing here reports a result; it produces the numbers the
rule applies to (two dialectical rounds, two replications, before any number is spoken)."""

import random

from cocoonml.attention import Attention
from cocoonml.schema import draw, step


def pack(artifacts, separator, length):
    """One context: artifacts' forms joined by the separator form, cut or padded to `length`.
    Returns (tokens, targets): targets are the next token, the last target the separator."""
    tokens = []
    for forms in artifacts:
        tokens.extend(forms)
        tokens.append(separator)
    tokens = tokens[:length]
    while len(tokens) < length:
        tokens.append(separator)
    targets = tokens[1:] + [separator]
    return tokens, targets


def _count_vectors(cut_count, total):
    """Every way `total` cuts can be spread over `cut_count` cut indices, as count vectors in
    lexicographic order."""
    if cut_count == 1:
        return [(total,)]
    return [(n,) + rest for n in range(total + 1) for rest in _count_vectors(cut_count - 1, total - n)]


def extension_code(extension, cut_count, resolution=1):
    """The key of an extension: its weight vector over the cuts taken in grains of 1/resolution,
    ranked injectively among all grain vectors by total then lexicographically, so two extensions
    share a key only when they are the same vector. With one-hot leaves and resolution 1 the grains
    are the counts per cut, the multiset code of 2026-10-08. (Before that the key was the sum of
    the cuts, which collapsed distinct extensions of different sizes: at sweep 05's size five
    extensions became three keys, 2.03 bits became 1.39.) A reader's key, not a model's token,
    since the extension is latent in the complete system."""
    grains = tuple(int(w * resolution) for w in extension)
    assert all(Fraction(w) * resolution == g for w, g in zip(extension, grains)), "extension off the grid"
    size = sum(grains)
    offset = sum(len(_count_vectors(cut_count, t)) for t in range(1, size))
    return offset + _count_vectors(cut_count, size).index(grains)


def extension_grain(table):
    """The grain the table's extensions need: 1 when every leaf is one-hot (integer counts per cut),
    else the table's resolution (graded leaves move by 1/resolution)."""
    from cocoonml.schema import _is_nonterminal
    for node in table.nodes:
        for e in node.sisters:
            for leaf in (e.first, e.second):
                if not _is_nonterminal(leaf) and leaf.weights is not None and any(w.denominator != 1 for w in leaf.weights):
                    return table.resolution
    return 1


def extension_vocabulary(cut_count, max_size, resolution=1):
    """How many keys a table needs: one per grain vector of up to `max_size` leaves at the grid."""
    return sum(len(_count_vectors(cut_count, t)) for t in range(1, max_size * resolution + 1))


def contexts(table, count, per_context, separator, length, rng, fill=False):
    """`count` contexts, each packing `per_context` draws from the table; with `fill`, draws until
    the next whole artifact does not fit, so every context is full and no position is rare in
    training (sweep 02 round one, item 3)."""
    if fill:
        return [filled(table, separator, length, rng)[:2] for _ in range(count)]
    return [pack([draw(table, rng)[0] for _ in range(per_context)], separator, length) for _ in range(count)]


def filled(table, separator, length, rng, extension_base=None, pairs=False):
    cut_count, resolution = table.cuts, extension_grain(table)
    """One full context: artifacts drawn until the next whole one does not fit, packed with form
    targets, or with meaning targets when `extension_base` is given, or as (string, extension)
    pairs in the input when `pairs` is set too. Returns (tokens, targets, number of artifacts)."""
    slot = 2 if pairs else 1  # a separator, and the extension token after it when pairs are in the input
    artifacts, used = [], 0
    while True:
        forms, cuts = draw(table, rng)
        if used + len(forms) + slot > length:
            break
        artifacts.append((forms, cuts))
        used += len(forms) + slot
    if extension_base is None:
        tokens, targets = pack([forms for forms, _ in artifacts], separator, length)
    elif pairs:
        tokens, targets = pack_pairs(artifacts, separator, length, extension_base, cut_count, resolution)
    else:
        tokens, targets = pack_meaning(artifacts, separator, length, extension_base, cut_count, resolution)
    return tokens, targets, len(artifacts)


def _poisson(rng, mean):
    """A count drawn around `mean` (Knuth's method; standard library only)."""
    if mean <= 0:
        return 0
    limit, k, product = math.exp(-mean), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k


def stream(table, rng, bound, lean, radius, separator, length, extension_base=None, pairs=False, steps_per_context=1):
    """Contexts produced while the index walks: each context is drawn from the current table, which
    then takes a number of steps of the bounded walk (`schema.step`) around `table`, the origin,
    within `radius` (None: a free walk). The number of steps between contexts is drawn around
    `steps_per_context` (a Poisson count: there is no clock in the index, steps are not ticks, so the
    count is a rate, not a metronome); 0 is the stationary table, which is what every sweep up to 05
    trained on. This is the training stream of the reality design: drift present in training, up to
    the training radius; readings beyond it are the test. Yields (tokens, targets, current table)
    without end."""
    current = table
    while True:
        tokens, targets, _ = filled(current, separator, length, rng, extension_base, pairs)
        yield tokens, targets, current
        for _ in range(_poisson(rng, steps_per_context)):
            current = step(current, rng, bound, lean, centre=table, radius=radius)


def train(model, table, steps, batch_size, per_context, separator, lr, rng):
    """Train the model on contexts from the table; returns the loss at each step."""
    losses = []
    for _ in range(steps):
        batch = contexts(table, batch_size, per_context, separator, model.length, rng)
        losses.append(model.train_step(batch, lr))
    return losses


def loss_by_position(model, table, count, per_context, separator, rng):
    """Mean cross-entropy per position over `count` contexts from `table`: the curve."""
    sums = [0.0] * model.length
    for tokens, targets in contexts(table, count, per_context, separator, model.length, rng):
        logits, _ = model.forward(tokens)
        for i, (lg, t) in enumerate(zip(logits, targets)):
            m = max(x.value for x in lg)
            total = sum(pow(2.718281828459045, x.value - m) for x in lg)
            sums[i] += -(lg[t].value - m) + __import__("math").log(total)
    return [s / count for s in sums]


def drifted(table, steps, bound, lean, rng, radius=None):
    """The table after `steps` moves of the index, each pulled toward the origin within `radius`
    (no pull when radius is None)."""
    origin = table
    for _ in range(steps):
        table = step(table, rng, bound, lean, centre=origin, radius=radius)
    return table


def run(table, model_seed, width, length, train_steps, batch_size, per_context, lr, drift_steps, bound, lean, eval_count, seed, radius=None):
    """The whole apparatus once: train on the table, read the curve on the table and on its
    drifted version. Returns (train_losses, curve_same, curve_drifted, the drifted table)."""
    rng = random.Random(seed)
    vocab = 1 + max(max((leaf.form for node in table.nodes for e in node.sisters for leaf in (e.first, e.second) if not isinstance(leaf, int)), default=0), 1) + 1
    separator = vocab - 1
    model = Attention(vocab=vocab, width=width, length=length, seed=model_seed)
    train_losses = train(model, table, train_steps, batch_size, per_context, separator, lr, rng)
    same = loss_by_position(model, table, eval_count, per_context, separator, rng)
    moved = drifted(table, drift_steps, bound, lean, rng, radius)
    after = loss_by_position(model, moved, eval_count, per_context, separator, rng)
    return train_losses, same, after, moved


# --- Meaning, not only form -----------------------------------------------------------------------
# Next-form prediction reads the orthographic side. The protocol side needs a target that depends
# on the cuts: at each separator the target is the artifact's EXTENSION, the sum of its cuts,
# offset past the forms so it is a token of its own. Then drift of the incidence (same form, new
# cut) is a measurable failure, and recovery of meaning through a drifted form is the task.


def pack_meaning(artifacts, separator, length, extension_base, cut_count=2, resolution=1):
    """Like pack, but the target at each separator is the artifact's extension token:
    extension_base + `extension_code` of its cuts. Elsewhere the target is the next form."""
    tokens, targets = [], []
    for forms, cuts in artifacts:
        if len(tokens) + len(forms) + 1 > length:
            break  # an artifact that does not fit whole is dropped: a cut artifact has no extension
        for i, f in enumerate(forms):
            tokens.append(f)
            targets.append(forms[i + 1] if i + 1 < len(forms) else separator)
        tokens.append(separator)
        targets.append(extension_base + extension_code(cuts, cut_count, resolution))
    while len(tokens) < length:
        tokens.append(separator)
        targets.append(separator)
    return tokens, targets


def contexts_meaning(table, count, per_context, separator, length, extension_base, rng, fill=False, pairs=False):
    if fill:
        return [filled(table, separator, length, rng, extension_base, pairs)[:2] for _ in range(count)]
    if pairs:
        return [pack_pairs([draw(table, rng) for _ in range(per_context)], separator, length, extension_base) for _ in range(count)]
    return [pack_meaning([draw(table, rng) for _ in range(per_context)], separator, length, extension_base) for _ in range(count)]


# --- Pairs in the input (sweep 03 round one, item 2) --------------------------------------------
# pack_meaning never puts an extension in the token sequence, so nothing in a window tells the model
# what an earlier artifact meant and the meaning loss is a trained lookup. With pairs, the extension
# token follows its separator as a token the model reads; the target at a separator is still the
# extension, so the loss at the k-th separator is the extension of the k-th string given k − 1
# (string, extension) pairs in context. The extension token's own target is whatever comes next.


def pack_pairs(artifacts, separator, length, extension_base, cut_count=2, resolution=1):
    """Tokens: forms, separator, extension token, per artifact; targets: the next token, so the
    extension at the separator. An artifact that does not fit whole with its pair is dropped."""
    tokens = []
    for forms, cuts in artifacts:
        if len(tokens) + len(forms) + 2 > length:
            break
        tokens.extend(forms)
        tokens.append(separator)
        tokens.append(extension_base + extension_code(cuts, cut_count, resolution))
    while len(tokens) < length:
        tokens.append(separator)
    targets = tokens[1:] + [separator]
    return tokens, targets


def _ranked(table, count, per_context, separator, length, extension_base, rng, fill, pairs=False):
    """Contexts with the number of artifacts each holds, and the rank every context reaches:
    with `fill`, the curves are reported only at ranks present in every context, since a later
    rank is otherwise read on the contexts whose earlier artifacts were short (sweep 02 round one,
    item 4). Without fill every rank is reported."""
    if fill:
        made = [filled(table, separator, length, rng, extension_base, pairs) for _ in range(count)]
        return [(t, g) for t, g, _ in made], min(n for _, _, n in made)
    if extension_base is None:
        return contexts(table, count, per_context, separator, length, rng), None
    return contexts_meaning(table, count, per_context, separator, length, extension_base, rng, pairs=pairs), None


def meaning_loss_at_separators(model, table, count, per_context, separator, length, extension_base, rng, fill=False, pairs=False):
    """Mean cross-entropy of the extension targets, in order of the separator's rank within the
    context (first artifact, second, ...): the recovery curve of meaning against k. With `pairs`
    the earlier pairs are in the input and the curve is an in-context readout."""
    import math

    sums, counts = {}, {}
    made, whole = _ranked(table, count, per_context, separator, length, extension_base, rng, fill, pairs)
    for tokens, targets in made:
        logits, _ = model.forward(tokens)
        rank = 0
        for i, (tok, lg, t) in enumerate(zip(tokens, logits, targets)):
            if tok == separator and t >= extension_base:
                m = max(x.value for x in lg)
                total = sum(math.exp(x.value - m) for x in lg)
                sums[rank] = sums.get(rank, 0.0) + (-(lg[t].value - m) + math.log(total))
                counts[rank] = counts.get(rank, 0) + 1
                rank += 1
    return [sums[r] / counts[r] for r in sorted(sums) if whole is None or r < whole]


def form_loss_by_rank(model, table, count, per_context, separator, rng, fill=False, extension_base=None, pairs=False):
    """Mean cross-entropy of next-FORM prediction grouped by artifact rank within the context,
    padding and separator targets excluded: the recovery curve of form against k. With `pairs`
    (and `extension_base`), contexts carry the pairs and the extension tokens are not form
    positions: the first form of an artifact is never a form target, in either layout."""
    import math

    sums, counts = {}, {}
    made, whole = _ranked(table, count, per_context, separator, model.length, extension_base if pairs else None, rng, fill, pairs)
    for tokens, targets in made:
        logits, _ = model.forward(tokens)
        rank = 0
        for i, (tok, lg, t) in enumerate(zip(tokens, logits, targets)):
            if tok == separator:
                rank += 1
                continue
            if extension_base is not None and tok >= extension_base:
                continue  # the extension token predicts the next artifact's first form, not a form inside one
            if t == separator:
                continue  # predicting the end of an artifact is not predicting a form
            m = max(x.value for x in lg)
            total = sum(math.exp(x.value - m) for x in lg)
            sums[rank] = sums.get(rank, 0.0) + (-(lg[t].value - m) + math.log(total))
            counts[rank] = counts.get(rank, 0) + 1
    return [sums[r] / counts[r] for r in sorted(sums) if counts[r] > 0 and (whole is None or r < whole)]


# --- Training to a plateau (sweep 02 round one, item 6; sweep 03, item 6) ----------------------
# A fixed step count left every cell still falling. The rule: train until the mean loss over the
# last `window` steps improves on the previous window's by less than `tolerance` of it, or `cap`.


def train_until_plateau(model, make_batch, lr, window=50, tolerance=0.01, cap=2000, evaluate=None, patience=1, minimum=0):
    """Train step by step on make_batch() until the plateau rule holds or the cap; returns
    (losses, evaluations), so the caller can report whether the cap or the rule ended training and
    the last readings. With `evaluate`, a callable returning the loss on a fixed held-out set, the
    rule reads it every `window` steps (sweep 04 round one, item 3: on the batch loss the rule stops
    at the batch noise); without it, the means of the last two windows of batch losses. The rule
    holds when `patience` consecutive readings each improve on the one before by less than
    `tolerance` of it, and is not consulted before `minimum` steps (sweep 05's first cells: one
    non-improving reading after a hundred steps ended training with the loss far from settled)."""
    losses, evaluations, stale = [], [], 0
    while len(losses) < cap:
        losses.append(model.train_step(make_batch(), lr))
        if not math.isfinite(losses[-1]):
            break  # the step overflowed: a diverged cell stops here instead of running to the cap on NaN (2026-10-09, the first derived-size cells)
        if len(losses) % window:
            continue
        if evaluate is not None:
            evaluations.append(evaluate())
            if len(evaluations) < 2:
                continue
            improved = evaluations[-1] < evaluations[-2] * (1 - tolerance)
        else:
            if len(losses) < 2 * window:
                continue
            last = sum(losses[-window:]) / window
            before = sum(losses[-2 * window : -window]) / window
            improved = last < before * (1 - tolerance)
        stale = 0 if improved else stale + 1
        if stale >= patience and len(losses) >= minimum:
            break
    return losses, evaluations


# --- The scrambled-pairs control (sweep 04 round one, item 2) -----------------------------------
# A drop of the meaning loss after the first artifact could be the pairs being read, or the first
# position being special. The control: the same context with the extension tokens of the input
# permuted among themselves, the targets untouched, so the model is asked for the true extension of
# each string while every earlier pair in its window is wrong. A curve that does not rise under
# scrambling did not read the pairs.


def scramble_extensions(tokens, rng, extension_base):
    """The tokens with their extension tokens permuted among themselves; forms and separators where
    they were. Returns a new list."""
    places = [i for i, t in enumerate(tokens) if t >= extension_base]
    values = [tokens[i] for i in places]
    rng.shuffle(values)
    out = list(tokens)
    for i, v in zip(places, values):
        out[i] = v
    return out
