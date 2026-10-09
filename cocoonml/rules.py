"""The rule table of docs/rules.md as functions: the parameter vector and the ensemble, the derived
floor, the sizes, a cell's command and its JSON line, the comparisons a cell's readings make, the
rows in their order, and the log entry. Pure: no I/O; runs/loop.py launches and writes."""
import json
import math
from dataclasses import dataclass, replace

SIGNS = {-1: "-", 0: "0", 1: "+"}
WHOLE_AT_THE_LENGTH_FLOOR = 8  # the L row: at least 8 whole artifacts per window
BITS_PER_PARAMETER = 3  # b of the cap row, an import, marked in docs/rules.md
BATCH_WINDOWS = 4  # the cap row: steps are batches of 4 windows


@dataclass(frozen=True)
class Vector:
    """docs/rules.md, "The parameter vector": θ = (w, ℓ, L, C, N, cap, η)."""

    width: int
    layers: int
    length: int
    corpus: int
    read_size: int
    cap: int
    lr: float


@dataclass(frozen=True)
class Ensemble:
    """docs/rules.md, "The parameter vector": the ensemble, fixed for a loop and moved by no row."""

    forms: int
    nonterminals: int
    resolution: int
    tables: int
    seed: int
    train_radii: tuple  # grains, one cell per value
    read_radii: tuple  # strings in runs/sweep7.py's form, e.g. ("0", "1", "2", "3", "free")
    steps_mean: float
    switch_mean: float
    patience: int
    minimum: int
    lean: str  # a fraction as text, e.g. "3/4"


def round_up_to(n, multiple):
    """The derived floor's w row: rounded up to a multiple (of 8)."""
    return math.ceil(n / multiple) * multiple


def corpus_floor_for(cap, mean_artifacts_per_window):
    """The derived floor's C row: at least 0.4 · cap · ā artifacts, ā the mean artifacts a window holds."""
    return math.ceil(0.4 * cap * mean_artifacts_per_window)


def derived_floor(distinct_strings, suffix_depth, mean_string_length, mean_artifacts_per_window, effective_count, cap):
    """The derived floor of docs/rules.md, every row: w from the distinct strings, ℓ from the suffix
    depth, L from the mean string length, the cap floor from the match's description against the
    bits a window carries (whole = 8 at the length floor), and the C floor at the given cap."""
    width = round_up_to(distinct_strings, 8)
    layers = suffix_depth + 1
    length = math.ceil(WHOLE_AT_THE_LENGTH_FLOOR * (mean_string_length + 1))
    whole_bits = WHOLE_AT_THE_LENGTH_FLOOR * math.log2(effective_count)
    windows = math.ceil(4 * (2 * width ** 2 * BITS_PER_PARAMETER) / whole_bits)
    return {
        "width": width,
        "layers": layers,
        "length": length,
        "cap_floor": math.ceil(windows / BATCH_WINDOWS),
        "corpus_floor": corpus_floor_for(cap, mean_artifacts_per_window),
    }


def sizes(width, layers):
    """Row R4's three sizes: (w, ℓ), (⌈1.5w⌉₈, ℓ + 1), (2w, ℓ + 2)."""
    return [(width, layers), (round_up_to(math.ceil(1.5 * width), 8), layers + 1), (2 * width, layers + 2)]


def next_size(largest_width, largest_layers):
    """Row R4's move: the size added, (⌈1.5 · largest⌉₈, ℓ + 1 of the largest)."""
    return (round_up_to(math.ceil(1.5 * largest_width), 8), largest_layers + 1)


def cell_name(train_radius, width, prefix="sweep-07"):
    """A sweep 07 cell's name (runs/README.md): <prefix>-r<train radius in grains>-w<width>."""
    return f"{prefix}-r{train_radius}-w{width}"


def rate_of(rates, vector):
    """The learning rate of the vector's size (decisions 40: the rate is per size, read off each size
    by R1's halving); `rates` maps (width, layers) to a rate; a size not in it takes the vector's."""
    return rates.get((vector.width, vector.layers), vector.lr)


def cell_command(python, script_path, name, ensemble, vector, train_radius, rates=None):
    """The argv of one cell in runs/sweep7.py's usage order, every element a string: the vector's
    seven numbers in their slots (the rate of the vector's size), the ensemble's in theirs."""
    lr = rate_of(rates or {}, vector)
    return [str(x) for x in (
        python, script_path, name,
        ensemble.forms, ensemble.nonterminals, ensemble.resolution,
        vector.width, vector.layers, vector.length,
        ensemble.tables, train_radius, ensemble.steps_mean, ensemble.switch_mean,
        vector.corpus, ",".join(ensemble.read_radii), vector.read_size, vector.cap, ensemble.seed,
        lr, ensemble.patience, ensemble.minimum, ensemble.lean,
    )]


def cell_at(cell, vector, train_radius, rates=None):
    """Whether a cell's JSON line was produced at this vector and training radius, read from its
    "args" (runs/sweep7.py's usage order; lr defaults to 0.1 when absent), so the loop resumes it
    instead of re-running it; cells of the same name at a moved vector do not match."""
    args = cell.get("args") or []
    if len(args) < 16:
        return False
    try:
        numbers = [int(args[i]) for i in (4, 5, 6, 8, 11, 13, 14)]
        lr = float(args[16]) if len(args) > 16 else 0.1
    except ValueError:
        return False
    at = [vector.width, vector.layers, vector.length, train_radius, vector.corpus, vector.read_size, vector.cap]
    return numbers == at and lr == rate_of(rates or {}, vector)


def parse_cell(log_text):
    """The cell's JSON line (runs/README.md: a line starting with {"run" at the end of its log): the
    dict of the last such line, or None when the log holds none that parses."""
    found = None
    for line in log_text.splitlines():
        if line.startswith('{"run"'):
            found = line
    if found is None:
        return None
    try:
        return json.loads(found)
    except ValueError:
        return None


def past_radius_keys(train_radius, read_radii):
    """Row R4's "past the training radius": the read keys numerically greater than it, and "free"."""
    return [key for key in read_radii if key == "free" or int(key) > train_radius]


def _sign(value, band):
    return 0 if abs(value) <= band else (1 if value > 0 else -1)


def readings(cell, train_radius, band=0.0):
    """Row R4's comparisons, each (value, sign) with |value| ≤ band giving sign 0: per read radius
    past the training radius, on the far bin by truth distance, the slope in k of the form curve
    from rank 0 to rank 3, its level at the last common rank against the inside (held) reading, and
    the order control's last value; and the held reading's own order control."""
    out = {}
    held = cell["held"]
    for key in past_radius_keys(train_radius, tuple(cell["points"])):
        point = cell["points"][key]
        bins = point["by_truth_distance"]
        if not bins:
            continue
        far = bins[-1]["form_by_rank"]
        n = len(far)
        if n >= 2:
            value = far[min(3, n - 1)] - far[0]
            out["slope_" + key] = (value, _sign(value, band))
        m = min(n, len(held["form_by_rank"])) - 1
        if m >= 0:
            value = far[m] - held["form_by_rank"][m]
            out["level_" + key] = (value, _sign(value, band))
        if point["order_control_by_rank"]:
            value = point["order_control_by_rank"][-1]
            out["control_" + key] = (value, _sign(value, band))
    if held["order_control_by_rank"]:
        value = held["order_control_by_rank"][-1]
        out["control_held"] = (value, _sign(value, band))
    return out


def r1_instability(cell):
    """Row R1 instability: the cell's loss is not finite (`diverged`), or a held-out evaluation
    exceeds the one before it by more than a factor of 2."""
    if cell.get("diverged"):
        return True, f"held at {cell['run']}: the loss is not finite"
    evaluations = cell["evaluations"]
    for i in range(1, len(evaluations)):
        if evaluations[i] > 2 * evaluations[i - 1]:
            return True, f"held at {cell['run']}: evaluation {i} more than doubled evaluation {i - 1}"
    return False, f"did not hold at {cell['run']}"


def r2_budget(cell, tolerance=0.01):
    """Row R2 budget: hit_cap and the last evaluation improved on the one before it by more than the tolerance."""
    evaluations = cell["evaluations"]
    if cell["hit_cap"] and len(evaluations) >= 2 and evaluations[-1] < evaluations[-2] * (1 - tolerance):
        return True, f"held at {cell['run']}: hit the cap with the last evaluation still improving past the tolerance"
    return False, f"did not hold at {cell['run']}"


def r3_k_range(cell):
    """Row R3 the k range: whole < 8 on the training windows, the held reading or any read radius."""
    short = []
    if cell["whole"] < WHOLE_AT_THE_LENGTH_FLOOR:
        short.append("corpus")
    if cell["held"]["whole"] < WHOLE_AT_THE_LENGTH_FLOOR:
        short.append("held")
    for key, point in cell["points"].items():
        if point["whole"] < WHOLE_AT_THE_LENGTH_FLOOR:
            short.append(key)
    if short:
        return True, f"held at {cell['run']}: whole below {WHOLE_AT_THE_LENGTH_FLOOR} at {', '.join(short)}"
    return False, f"did not hold at {cell['run']}"


def r4_disagree(cells_by_size, train_radius, band=0.0):
    """Row R4 replication by size: over the sizes run at this vector (smallest first), a comparison
    present in all of them has signs that differ; the detail names the keys and their signs."""
    signs = [{key: s for key, (_, s) in readings(cell, train_radius, band).items()} for cell in cells_by_size]
    keys = [key for key in signs[0] if all(key in s for s in signs)] if signs else []
    differing = [key for key in keys if len({s[key] for s in signs}) > 1]
    names = ", ".join(cell["run"] for cell in cells_by_size)
    if differing:
        detail = "; ".join(f"{key} ({', '.join(SIGNS[s[key]] for s in signs)})" for key in differing)
        return True, f"held over {names}: {detail}"
    return False, f"did not hold over {names}: {', '.join(keys) if keys else 'no comparison in common'} agree"


def host_exceeds(last_seconds, width, layers, next_width, next_layers, limit_seconds=3600.0):
    """The host row: the next size's time, estimated from the last cell's seconds scaled by
    (w'/w)² · (ℓ'/ℓ), exceeds the limit per cell."""
    estimate = last_seconds * (next_width / width) ** 2 * (next_layers / layers)
    where = f"({next_width}, {next_layers}) estimated at {estimate:.0f} s per cell from {last_seconds:.0f} s at ({width}, {layers})"
    if estimate > limit_seconds:
        return True, f"held: {where}, over {limit_seconds:.0f} s"
    return False, f"did not hold: {where}"


def sizes_that_fired_r1(cells):
    """The (width, layers) of every cell R1 holds at: the sizes whose rate the loop halves."""
    return sorted({(cell["width"], cell["layers"]) for cell in cells.values() if r1_instability(cell)[0]})


def apply_rows(cells, train_radii, vector, mean_artifacts_per_window, band=0.0, tolerance=0.01):
    """The rows in the order of docs/rules.md over every cell at the vector, the first that holds
    firing: R1 names the cells whose size's rate halves (the vector stands); R2 doubles the cap and raises C to its floor at the new cap; R3 lengthens
    L to ⌈1.5 L⌉; R4 (over each training radius's sizes) returns None for the vector since the
    size list moves, in the loop; "stop" returns the vector unchanged. cells is
    {(train_radius, width): cell}; the result is (row, next vector or None, detail)."""
    ordered = [cells[key] for key in sorted(cells)]
    fired = [(key, r1_instability(cells[key])) for key in sorted(cells)]
    fired = [(key, detail) for key, (fires, detail) in fired if fires]
    if fired:
        # R1 is per cell and moves that size's rate (decisions 40): the vector stands, the rates move;
        # the detail names every cell that fired, and the loop halves the rate of each of their sizes
        return "R1", vector, "; ".join(detail for _, detail in fired)
    for cell in ordered:
        fires, detail = r2_budget(cell, tolerance)
        if fires:
            cap = 2 * vector.cap
            corpus = max(vector.corpus, corpus_floor_for(cap, mean_artifacts_per_window))
            return "R2", replace(vector, cap=cap, corpus=corpus), detail
    for cell in ordered:
        fires, detail = r3_k_range(cell)
        if fires:
            return "R3", replace(vector, length=math.ceil(1.5 * vector.length)), detail
    for train_radius in train_radii:
        by_size = [cells[key] for key in sorted(cells) if key[0] == train_radius]
        fires, detail = r4_disagree(by_size, train_radius, band)
        if fires:
            return "R4", None, detail
    return "stop", vector, "no row fires"


def _vector_text(vector):
    return f"({vector.width}, {vector.layers}, {vector.length}, {vector.corpus}, {vector.read_size}, {vector.cap}, {vector.lr})"


def log_entry(stamp, vector, sizes_run, cells, row, detail, next_vector):
    """docs/rules.md, "The log entry": one line with the stamp, the vector, the sizes run with each
    cell's seconds, the row that fired with the comparison it made, and the next vector or "stop";
    no reading's number."""
    run = ", ".join(f"{cells[key]['run']} {cells[key]['seconds']} s" for key in sorted(cells))
    sized = ", ".join(f"({w}, {l})" for w, l in sizes_run)
    following = "stop" if next_vector is None else _vector_text(next_vector)
    return f"- {stamp} · vector {_vector_text(vector)} · sizes {sized}: {run} · row {row}: {detail} · next: {following}"
