import math

from cocoonml.autograd import Autograd


def softmax_cross_entropy_composed(scores, target):
    """The loss built from primitive operations only. Its backward is the chain rule
    with nothing simplified: the check the fused version is measured against."""
    exps = [s.exp() for s in scores]
    total = exps[0]
    for e in exps[1:]:
        total = total.add(e)
    return -(exps[target] / total).log()


def softmax_cross_entropy(scores, target):
    """The fused loss: one node over all n scores, with the gradient written in closed form.
    The closed form is the derivation (docs/concepts/02-softmax-cross-entropy.md); until it is
    written, backward raises."""
    values = [s.value for s in scores]
    m = max(values)  # shift for stability; softmax is invariant to it
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    probs = [e / total for e in exps]
    out = Autograd(-math.log(probs[target]), tuple(scores), "softmax_xent")

    def _backward():
        # d loss / d score_i, in closed form, for every i: the line to derive.
        raise NotImplementedError("the gradient of softmax cross-entropy is not written yet")

    out._backward = _backward
    return out
