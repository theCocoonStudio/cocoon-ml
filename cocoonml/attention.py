"""The model side: one layer of causal softmax attention on the scalar autograd, with a probe.

Tiny by design: vocabulary V (forms, 0 the empty form), embedding width d, sequence length n.
Every weight is an Autograd scalar so the gradients come from the engine in cocoonml/autograd.py
and nothing else. The probe returns the attended vectors per position, the activations a reading
of the index would have to be found in.
"""

import random

from cocoonml.autograd import Autograd
from cocoonml.loss import softmax_cross_entropy_composed


def _matrix(rows, cols, rng, scale):
    return [[Autograd(rng.uniform(-scale, scale)) for _ in range(cols)] for _ in range(rows)]


def _dot(a, b):
    out = a[0] * b[0]
    for x, y in zip(a[1:], b[1:]):
        out = out.add(x * y)
    return out


def _matvec(matrix, vector):
    # matrix: rows x len(vector) ; returns a list of rows
    return [_dot(row, vector) for row in matrix]


def _softmax(scores):
    m = max(s.value for s in scores)
    exps = [s.add(-m).exp() for s in scores]
    total = exps[0]
    for e in exps[1:]:
        total = total.add(e)
    return [e / total for e in exps]


class Attention:
    def __init__(self, vocab, width, length, seed):
        rng = random.Random(seed)
        scale = 0.5
        self.vocab, self.width, self.length = vocab, width, length
        self.embed = _matrix(vocab, width, rng, scale)
        self.position = _matrix(length, width, rng, scale)
        self.query = _matrix(width, width, rng, scale)
        self.key = _matrix(width, width, rng, scale)
        self.value = _matrix(width, width, rng, scale)
        self.readout = _matrix(vocab, width, rng, scale)

    def parameters(self):
        for block in (self.embed, self.position, self.query, self.key, self.value, self.readout):
            for row in block:
                yield from row

    def forward(self, tokens):
        """Logits per position over the vocabulary, and the attended vectors (the probe)."""
        xs = [[e.add(p) for e, p in zip(self.embed[t], self.position[i])] for i, t in enumerate(tokens)]
        qs = [_matvec(self.query, x) for x in xs]
        ks = [_matvec(self.key, x) for x in xs]
        vs = [_matvec(self.value, x) for x in xs]
        attended, logits = [], []
        for i in range(len(tokens)):
            scores = [_dot(qs[i], ks[j]) for j in range(i + 1)]  # causal: positions up to i
            weights = _softmax(scores)
            mixed = [weights[0] * vs[0][c] for c in range(self.width)]
            for j in range(1, i + 1):
                mixed = [mixed[c].add(weights[j] * vs[j][c]) for c in range(self.width)]
            attended.append(mixed)
            logits.append(_matvec(self.readout, mixed))
        return logits, attended

    def loss(self, tokens, targets):
        """Mean cross-entropy of predicting targets[i] from positions 0..i."""
        logits, _ = self.forward(tokens)
        total = softmax_cross_entropy_composed(logits[0], targets[0])
        for lg, t in zip(logits[1:], targets[1:]):
            total = total.add(softmax_cross_entropy_composed(lg, t))
        return total / float(len(tokens))

    def probe(self, tokens):
        """The attended vectors as plain numbers, per position: what a probe reads."""
        _, attended = self.forward(tokens)
        return [[a.value for a in row] for row in attended]

    def train_step(self, batch, lr):
        """One gradient step on a batch of (tokens, targets); returns the batch loss."""
        for p in self.parameters():
            p._derivative = 0.0
        total = None
        for tokens, targets in batch:
            l = self.loss(tokens, targets)
            total = l if total is None else total.add(l)
        total = total / float(len(batch))
        total.backward()
        for p in self.parameters():
            p.value -= lr * p._derivative
        return total.value
