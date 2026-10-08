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


def contexts(table, count, per_context, separator, length, rng):
    """`count` contexts, each packing `per_context` draws from the table."""
    return [pack([draw(table, rng)[0] for _ in range(per_context)], separator, length) for _ in range(count)]


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


def drifted(table, steps, bound, lean, rng):
    """The table after `steps` moves of the index."""
    for _ in range(steps):
        table = step(table, rng, bound, lean)
    return table


def run(table, model_seed, width, length, train_steps, batch_size, per_context, lr, drift_steps, bound, lean, eval_count, seed):
    """The whole apparatus once: train on the table, read the curve on the table and on its
    drifted version. Returns (train_losses, curve_same, curve_drifted, delta_steps)."""
    rng = random.Random(seed)
    vocab = 1 + max(max((leaf.form for node in table.nodes for e in node.sisters for leaf in (e.first, e.second) if not isinstance(leaf, int)), default=0), 1) + 1
    separator = vocab - 1
    model = Attention(vocab=vocab, width=width, length=length, seed=model_seed)
    train_losses = train(model, table, train_steps, batch_size, per_context, separator, lr, rng)
    same = loss_by_position(model, table, eval_count, per_context, separator, rng)
    moved = drifted(table, drift_steps, bound, lean, rng)
    after = loss_by_position(model, moved, eval_count, per_context, separator, rng)
    return train_losses, same, after, moved
