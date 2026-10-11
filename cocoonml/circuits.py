"""The circuits reader (docs/literature.md, entries 2 and 3): two scores on a model's attention
weights, read on a copy of the parameters after training. The previous-token score: the attention
each position puts on the position before it. The induction score (Olsson et al., 2022): for each
position whose token occurred earlier in the sequence, the attention it puts on the position after
the most recent earlier occurrence, the place an induction head reads what followed. Both per layer,
averaged over the positions that qualify. Numbers for the log; what they mean is the rounds' question."""
import numpy as np


def previous_token_score(weights):
    """Mean attention from position i to position i - 1 over i >= 1; None for a sequence of one."""
    n = weights.shape[0]
    if n < 2:
        return None
    return float(np.mean([weights[i, i - 1] for i in range(1, n)]))


def induction_score(tokens, weights, exclude=()):
    """Mean attention from position i to position j + 1, j the most recent earlier position with the
    same token as i, over the positions i that have such a j with j + 1 < i (an adjacent repeat has
    nothing to read); positions whose token is in `exclude` (separators) are not queries. None when
    no position qualifies."""
    scores = []
    last = {}
    for i, token in enumerate(tokens):
        j = last.get(token)
        if j is not None and j + 1 < i and token not in exclude:
            scores.append(weights[i, j + 1])
        last[token] = i
    return float(np.mean(scores)) if scores else None


def circuit_scores(model, sequences, exclude=()):
    """Both scores per layer, averaged over the sequences that qualify: {"previous_token": [per layer],
    "induction": [per layer]}, None where no sequence qualified."""
    previous, induction = None, None
    for tokens in sequences:
        layers = model.attention(tokens)
        if previous is None:
            previous = [[] for _ in layers]
            induction = [[] for _ in layers]
        for depth, weights in enumerate(layers):
            p = previous_token_score(weights)
            if p is not None:
                previous[depth].append(p)
            s = induction_score(tokens, weights, exclude)
            if s is not None:
                induction[depth].append(s)
    if previous is None:
        return {"previous_token": [], "induction": []}
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"previous_token": [mean(xs) for xs in previous], "induction": [mean(xs) for xs in induction]}
