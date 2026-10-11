"""The rule table of docs/rules.md as functions: the parameter vector and the ensemble, the derived
floor, the sizes, a cell's command and its JSON line, the comparisons a cell's readings make, the
rows in their order, and the log entry. Pure: no I/O; runs/loop.py launches and writes."""
import json
import math
import statistics
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


def host_exceeds(last_seconds, width, layers, next_width, next_layers, limit_seconds=4 * 3600.0):
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
    # R1 no longer moves anything (decisions 53: the ladder of rates is the search, a divergence is a
    # reading); the cells it holds at are named in the detail of whatever fires, and the rate stands.
    unstable = "; ".join(detail for _, (fires, detail) in ((key, r1_instability(cells[key])) for key in sorted(cells)) if fires)
    noted = f" (R1 holds, a reading: {unstable})" if unstable else ""
    for cell in ordered:
        fires, detail = r2_budget(cell, tolerance)
        if fires:
            cap = 2 * vector.cap
            corpus = max(vector.corpus, corpus_floor_for(cap, mean_artifacts_per_window))
            return "R2", replace(vector, cap=cap, corpus=corpus), detail + noted
    for cell in ordered:
        fires, detail = r3_k_range(cell)
        if fires:
            return "R3", replace(vector, length=math.ceil(1.5 * vector.length)), detail + noted
    for train_radius in train_radii:
        by_size = [cells[key] for key in sorted(cells) if key[0] == train_radius]
        fires, detail = r4_disagree(by_size, train_radius, band)
        if fires:
            return "R4", None, detail + noted
    return "stop", vector, "no row fires" + noted


def _vector_text(vector):
    return f"({vector.width}, {vector.layers}, {vector.length}, {vector.corpus}, {vector.read_size}, {vector.cap}, {vector.lr})"


def log_entry(stamp, vector, sizes_run, cells, row, detail, next_vector, rates=None):
    """docs/rules.md, "The log entry": one line with the stamp, the vector, the sizes run with each
    cell's seconds, the rates per size when they differ from the vector's (decisions 53: the rate each
    size ran at is part of the vector, a debt Buridan named), the row that fired with the comparison
    it made, and the next vector or "stop"; no reading's number."""
    run = ", ".join(f"{cells[key]['run']} {cells[key]['seconds']} s" for key in sorted(cells))
    sized = ", ".join(f"({w}, {l})" for w, l in sizes_run)
    following = "stop" if next_vector is None else _vector_text(next_vector)
    per_size = ""
    if rates:
        per_size = " · rates per size " + ", ".join(f"({w}, {l}) {rate}" for (w, l), rate in sorted(rates.items()))
    return f"- {stamp} · vector {_vector_text(vector)} · sizes {sized}: {run}{per_size} · row {row}: {detail} · next: {following}"


# --- Decisions 53, the run's row at the statistical layer: its predicates and tests, pure ---------
# The run of twenty-four cells (three sizes, two training radii, a ladder of four rates) runs every
# cell to its cap or its escape with no stopping rule; these functions are applied to the record
# after it. Every constant below is the row's and is named there; nothing here reads a file.

ENVELOPE = 0.0378  # the null's envelope: the largest departure any of the six flat cells of 07c and 07d showed (r2-w192 at 200); recomputed on the JSON lines 2026-10-10 22:19
WINDOW = 50  # the held evaluation every fifty steps (runs/sweep7.py)


def departures(evaluations):
    """Each held evaluation's departure below the running median of the evaluations before it (None
    for the first, which has no reference). The reference is the plateau's level as the trajectory
    reports it, so the first evaluation's offset above the plateau, and that offset's dependence on
    the rate, drop out by construction (decisions 53, fourth form (1))."""
    out = [None]
    for j in range(1, len(evaluations)):
        out.append(statistics.median(evaluations[:j]) - evaluations[j])
    return out


def envelope(flat_trajectories):
    """The null's envelope over the flat cells' held trajectories: the largest departure any showed;
    None when no trajectory has two evaluations."""
    found = [d for ev in flat_trajectories for d in departures(ev) if d is not None]
    return max(found) if found else None


def onset(evaluations, envelope=ENVELOPE, persistence=2, window=WINDOW):
    """The onset step: the first of `persistence` consecutive evaluations whose departure exceeds the
    envelope; a step's window is the evaluation index of the window holding it, so evaluation i
    (counted from 1) is step window · i (the window convention, third form (1)). None if never: a
    cell with no onset has no best copy and gives the null its marginal copy only."""
    dep = departures(evaluations)
    for j in range(1, len(dep) - persistence + 1):
        if all(dep[j + k] > envelope for k in range(persistence)):
            return window * (j + 1)
    return None


def second_descents(evaluations, envelope=ENVELOPE, plateau=6, persistence=2, window=WINDOW, after=None):
    """Every step at which a second descent begins (the predicate of second_descent, every qualifying
    position rather than the first), in order; empty if none (Methuselah, 2026-10-11, round one on the
    record: a cell whose first second descent is early could never read "within" a later window)."""
    out = []
    for j in range(plateau, len(evaluations) - persistence + 1):
        if after is not None and window * (j - plateau + 1) <= after:
            continue
        level = statistics.median(evaluations[j - plateau : j])
        if any(abs(e - level) > envelope for e in evaluations[j - plateau : j]):
            continue
        if all(level - evaluations[j + k] > envelope for k in range(persistence)):
            out.append(window * (j + 1))
    return out


def second_descent(evaluations, envelope=ENVELOPE, plateau=6, persistence=2, window=WINDOW, after=None):
    """Prediction (4)'s predicate (third form (7)): a plateau of `plateau` evaluations, each within the
    envelope of their median (held, not still falling: the clause that makes the descent a second
    one, a choice marked here), followed by `persistence` consecutive evaluations below that median
    by more than the envelope; returns the step of the first of them, or None. With `after` (the
    cell's onset step), only plateaus whose first evaluation lies after it count, so the marginal
    plateau before the onset is not read as a plateau and the onset's own descent is not a second one."""
    found = second_descents(evaluations, envelope, plateau, persistence, window, after)
    return found[0] if found else None


def concordance(escapes):
    """Test (1), the ladder per size: `escapes` maps each rate to its escape step, or None for none
    within the cap. Over the pairs with at least one escape (two cap ties are excluded, as are two
    equal escape steps), a pair is concordant when the higher rate escaped earlier. Returns
    (concordant, discordant); decision (d) fires on a discordant pair."""
    rates = sorted(escapes)
    concordant = discordant = 0
    for i, low in enumerate(rates):
        for high in rates[i + 1 :]:
            at_low, at_high = escapes[low], escapes[high]
            if at_low is None and at_high is None:
                continue
            if at_low is None or (at_high is not None and at_high < at_low):
                concordant += 1
            elif at_high is None or at_high > at_low:
                discordant += 1
    return concordant, discordant


def rank_sign(value, null_values):
    """The rank rule (second form (iv)): a best copy's reading counts where it lies outside the
    null's range at its key and radius, on either side; the sign is the side of the null, +1 above
    it, -1 below, 0 inside (abstains). An empty null counts nothing."""
    if not null_values:
        return 0
    if value > max(null_values):
        return 1
    if value < min(null_values):
        return -1
    return 0


def differ_pairwise(signs):
    """R4's "differ" as the second form restates it, for one key over the sizes' signs: zeros abstain;
    the key enters where it counts at two sizes or more; it differs when any counting pair differs.
    Returns None (does not enter), False (agree) or True (differ)."""
    counting = [s for s in signs if s]
    if len(counting) < 2:
        return None
    return len(set(counting)) > 1


def crossing(width, whole, effective_count):
    """Decisions 48, the one-times crossing at the cell's own whole: the step at which the windows
    seen carry as many bits as the match's parameters, 2w² · b against whole · log₂(effective string
    count) per window, in batches of 4 (derived_floor's cap row at whole 8); 311, 699 and 1 243 at
    whole 9 for the three sizes."""
    return math.ceil(2 * width ** 2 * BITS_PER_PARAMETER / (BATCH_WINDOWS * whole * math.log2(effective_count)))


def cap_for(width, whole, effective_count, corpus_windows, crossings=8, passes=10):
    """Decisions 53, the cap per size: the smaller of `crossings` times the crossing and `passes`
    passes of the corpus (each window drawn at most ten times: corpus_windows · passes / 4 steps,
    rounded down); 2 488, 5 592 and 9 127 at the three sizes on the record's corpora."""
    return min(crossings * crossing(width, whole, effective_count), corpus_windows * passes // BATCH_WINDOWS)
