# 01 Autograd

A number that remembers how it was made.

Each value stores the inputs that produced it and the local derivative of the operation that did. Backward is the chain rule walked in reverse topological order from the output, accumulating into each input's gradient.

That is the whole engine under every neural net. About 60 lines.

What the code must do:

- `Value(x)` holds a scalar, its `grad`, its parents and the op.
- `+`, `*`, `**`, and one nonlinearity (`tanh` or `relu`), each recording its local derivative.
- `backward()` on any value: topological sort, then walk it in reverse and accumulate.
- Gradients accumulate, so a value used twice gets both contributions.

Check it against finite differences: nudge an input by `h`, the output changes by about `grad * h`.
