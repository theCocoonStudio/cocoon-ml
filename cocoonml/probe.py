"""A linear probe: ridge regression in closed form, pure Python, for reading a variable off the
attended vectors. The consistency-tracking claim is tested by whether a probe fitted on the
activations predicts the drift the window carries; this is the ruler for that reading."""


def _solve(a, b):
    """Gaussian elimination with partial pivoting for a square system a x = b."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            raise ValueError("singular system")
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(xs, ys, ridge=1e-3):
    """Weights w and bias b minimising Σ (w·x + b − y)² + ridge·|w|²."""
    d = len(xs[0])
    xa = [list(x) + [1.0] for x in xs]
    ata = [[sum(r[i] * r[j] for r in xa) for j in range(d + 1)] for i in range(d + 1)]
    for i in range(d):
        ata[i][i] += ridge
    atb = [sum(r[i] * y for r, y in zip(xa, ys)) for i in range(d + 1)]
    w = _solve(ata, atb)
    return w[:d], w[d]


def predict(w, b, x):
    return sum(wi * xi for wi, xi in zip(w, x)) + b


def r_squared(w, b, xs, ys):
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((predict(w, b, x) - y) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
