"""The scalar model of cocoonml/attention.py on numpy arrays, with the backward in closed form.

Same architecture, same weights for the same constructor arguments (drawn from the same
random.Random in the same order), same loss and gradients to rounding: the scalar model is the
oracle, this is the apparatus that runs in minutes where the scalar takes hours. No autograd:
every gradient below is the derivative written out, checked against the scalar in
tests/test_array_attention.py.
"""

import math
import random

import numpy as np


def _matrix(rows, cols, rng, scale):
    # the scalar's loop nest, rows then cols, so the floats come out identical
    return np.array([[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)], dtype=float)


def _ordered(source):
    """The blocks of `source` (a dict keyed like the attributes) in the scalar's parameter order."""
    out = [source[name] for name in ("embed", "position", "query", "key", "value", "readout")]
    if source["within"] is not None:
        out.append(source["within"])
    for q, k, v in source["more"]:
        out.extend((q, k, v))
    return out


class ArrayAttention:
    def __init__(self, vocab, width, length, seed, separator=None, layers=1, scale=0.5):
        """`scale`: every weight is drawn uniform in ±scale. 0.5 is the scalar model's (the oracle
        tests and every sweep up to 07's smoke); the runs at the derived width pass √(3/width), the
        one scale with no free constant that keeps the stream's variance through a layer and the
        attention scores of order one (docs/decisions.md 32: at 0.5 and width 96 the scores have a
        standard deviation near 20 and the softmax is saturated at initialisation)."""
        rng = random.Random(seed)
        self.vocab, self.width, self.length = vocab, width, length
        if scale == "derived":
            # The scale that reads off every count of the model (2026-10-09, after the nine-layer
            # overflow and Izzy's "look for your own choices"): a uniform ±s entry has variance s²/3.
            # Embeddings: three are summed, so each at variance 1/(3w) and the stream starts at 1/w.
            # Queries and keys: the scores sum w products with no division, so entries at variance
            # 1/√w give scores of variance one. Values: ℓ − 1 layers add to the stream (the last
            # layer's attended vector is read out, not added; forward below), so entries at variance
            # 1/((ℓ − 1)w) keep the stream within a factor of e of its start, 1/w when nothing adds.
            # The first version divided by ℓ, one more than the layers that add (Buridan's list,
            # 2026-10-09; decisions 44). Readout: 1/w.
            embeddings = 3 if separator is not None else 2
            s_embed = math.sqrt(3.0 / (embeddings * width))
            s_qk = math.sqrt(3.0 / math.sqrt(width))
            s_value = math.sqrt(3.0 / (max(layers - 1, 1) * width))
            s_read = math.sqrt(3.0 / width)
        else:
            s_embed = s_qk = s_value = s_read = scale
        self.embed = _matrix(vocab, width, rng, s_embed)
        self.position = _matrix(length, width, rng, s_embed)
        self.query = _matrix(width, width, rng, s_qk)
        self.key = _matrix(width, width, rng, s_qk)
        self.value = _matrix(width, width, rng, s_value)
        self.readout = _matrix(vocab, width, rng, s_read)
        self.separator = separator
        self.within = _matrix(length, width, rng, s_embed) if separator is not None else None
        self.layers = layers
        self.more = [tuple(_matrix(width, width, rng, s) for s in (s_qk, s_qk, s_value)) for _ in range(layers - 1)]
        self.grads = None

    # ----- parameters -----

    def blocks(self):
        """The weight arrays in the scalar's parameter order (the arrays themselves, not copies)."""
        return _ordered(
            {
                "embed": self.embed,
                "position": self.position,
                "query": self.query,
                "key": self.key,
                "value": self.value,
                "readout": self.readout,
                "within": self.within,
                "more": self.more,
            }
        )

    def flat_parameters(self):
        """Every weight as a Python float, row-major per block, blocks in the scalar's order."""
        return np.concatenate([block.ravel() for block in self.blocks()]).tolist()

    def _grad_blocks(self):
        if self.grads is None:
            raise RuntimeError("no gradients yet: call loss_and_gradients or train_step first")
        return _ordered(self.grads)

    def flat_gradients(self):
        """The gradients of the last backward, flattened like flat_parameters."""
        return np.concatenate([grad.ravel() for grad in self._grad_blocks()]).tolist()

    def _zero_grads(self):
        return {
            "embed": np.zeros_like(self.embed),
            "position": np.zeros_like(self.position),
            "query": np.zeros_like(self.query),
            "key": np.zeros_like(self.key),
            "value": np.zeros_like(self.value),
            "readout": np.zeros_like(self.readout),
            "within": None if self.within is None else np.zeros_like(self.within),
            "more": [tuple(np.zeros_like(w) for w in triple) for triple in self.more],
        }

    def _layer_weights(self):
        return [(self.query, self.key, self.value)] + list(self.more)

    # ----- forward -----

    def offsets(self, tokens):
        """Position since the last separator, the separator itself at zero; before any separator,
        the absolute position."""
        out, last = [], None
        for i, t in enumerate(tokens):
            if t == self.separator:
                last = i
            out.append(i if last is None else i - last)
        return out

    def _offsets_array(self, tokens):
        return np.array([self.offsets(row.tolist()) for row in tokens], dtype=int)

    def _forward_batch(self, tokens):
        """Logits (B x n x vocab), the last layer's attended vectors (B x n x width), and the cache
        for the backward: per layer the stream x, q, k, v and the attention weights A."""
        tokens = np.asarray(tokens, dtype=int)
        _, n = tokens.shape
        x = self.embed[tokens] + self.position[:n]
        if self.within is not None:
            x = x + self.within[self._offsets_array(tokens)]
        mask = np.tril(np.ones((n, n), dtype=bool))
        cache = []
        attended = None
        for depth, (Q, K, V) in enumerate(self._layer_weights()):
            q = x @ Q.T
            k = x @ K.T
            v = x @ V.T
            scores = np.where(mask, q @ np.swapaxes(k, -1, -2), -np.inf)
            scores = scores - scores.max(axis=-1, keepdims=True)
            exps = np.exp(scores)
            weights = exps / exps.sum(axis=-1, keepdims=True)
            attended = weights @ v
            cache.append((x, q, k, v, weights))
            if depth + 1 < self.layers:
                x = x + attended
        logits = attended @ self.readout.T
        return logits, attended, cache

    def forward(self, tokens):
        """Logits per position over the vocabulary (n x vocab), and the last layer's attended
        vectors (n x width)."""
        logits, attended, _ = self._forward_batch(np.array([tokens], dtype=int))
        return logits[0], attended[0]

    @staticmethod
    def _cross_entropy(logits, targets):
        """-log softmax(logits)[target] along the last axis, stably; any leading shape."""
        m = logits.max(axis=-1, keepdims=True)
        log_total = np.log(np.exp(logits - m).sum(axis=-1)) + m[..., 0]
        picked = np.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return log_total - picked

    def position_losses(self, tokens, targets):
        """The cross-entropy of predicting targets[i] at position i, per position."""
        logits, _ = self.forward(tokens)
        return self._cross_entropy(logits, np.asarray(targets, dtype=int)).tolist()

    def loss(self, tokens, targets):
        """Mean cross-entropy of predicting targets[i] from positions 0..i."""
        losses = self.position_losses(tokens, targets)
        return sum(losses) / float(len(losses))

    def probe(self, tokens):
        """The attended vectors as plain numbers, per position: what a probe reads."""
        _, attended = self.forward(tokens)
        return attended.tolist()

    # ----- backward -----

    def _backward(self, tokens, targets, logits, attended, cache, denominator, grads):
        """Accumulate into `grads` the gradient of (1/denominator) * (1/n) * sum of the
        cross-entropies of this B x n batch; `denominator` is the size of the whole batch."""
        batch, n = tokens.shape
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exps = np.exp(shifted)
        dlogits = exps / exps.sum(axis=-1, keepdims=True)
        dlogits[np.arange(batch)[:, None], np.arange(n)[None, :], targets] -= 1.0
        dlogits /= denominator * n
        grads["readout"] += np.einsum("bik,bid->kd", dlogits, attended)
        da = dlogits @ self.readout
        weights = self._layer_weights()
        dx = None
        for depth in reversed(range(self.layers)):
            x, q, k, v, A = cache[depth]
            Q, K, V = weights[depth]
            if depth + 1 < self.layers:
                da = dx  # the residual add passes the stream's gradient to this layer's attended vectors
            dv = np.swapaxes(A, -1, -2) @ da
            dA = da @ np.swapaxes(v, -1, -2)
            dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))  # softmax backward; masked entries have A = 0
            dq = dS @ k
            dk = np.swapaxes(dS, -1, -2) @ q
            dQ = np.einsum("bir,bic->rc", dq, x)
            dK = np.einsum("bir,bic->rc", dk, x)
            dV = np.einsum("bir,bic->rc", dv, x)
            dx_attention = dq @ Q + dk @ K + dv @ V
            dx = dx_attention if depth + 1 == self.layers else dx + dx_attention
            if depth == 0:
                grads["query"] += dQ
                grads["key"] += dK
                grads["value"] += dV
            else:
                gq, gk, gv = grads["more"][depth - 1]
                gq += dQ
                gk += dK
                gv += dV
        np.add.at(grads["embed"], tokens, dx)
        grads["position"][:n] += dx.sum(axis=0)
        if self.within is not None:
            np.add.at(grads["within"], self._offsets_array(tokens), dx)

    def loss_and_gradients(self, batch):
        """The scalar train_step's loss, (1/B) sum over contexts of the mean cross-entropy, and its
        gradients in self.grads (one array per block, keyed like the attributes, summed over the
        batch). Equal-length contexts run as one batched pass; otherwise one at a time."""
        size = len(batch)
        grads = self._zero_grads()
        if len({len(tokens) for tokens, _ in batch}) == 1:
            tokens = np.array([t for t, _ in batch], dtype=int)
            targets = np.array([g for _, g in batch], dtype=int)
            logits, attended, cache = self._forward_batch(tokens)
            loss = float(self._cross_entropy(logits, targets).mean())
            self._backward(tokens, targets, logits, attended, cache, size, grads)
        else:
            loss = 0.0
            for tokens, targets in batch:
                tokens = np.array([tokens], dtype=int)
                targets = np.array([targets], dtype=int)
                logits, attended, cache = self._forward_batch(tokens)
                loss += float(self._cross_entropy(logits, targets).mean()) / size
                self._backward(tokens, targets, logits, attended, cache, size, grads)
        self.grads = grads
        return loss

    def train_step(self, batch, lr):
        """One gradient step on a batch of (tokens, targets); returns the batch loss."""
        loss = self.loss_and_gradients(batch)
        for block, grad in zip(self.blocks(), self._grad_blocks()):
            block -= lr * grad
        return float(loss)
