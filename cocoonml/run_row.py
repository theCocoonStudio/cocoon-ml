"""Decisions 53 applied to a record: the run's tests, written before the run, over the JSON lines of
its cells after it (runs/sweep7.py in mode "cap"), each outcome mapped to the decision the row names.
Pure: no I/O; runs/row53.py loads the logs and prints the report. Nothing here reads a curve as a
result: the output names which decision each outcome maps to, and the numbers it rests on, in the log."""
import math
import statistics
from collections import defaultdict

from cocoonml.rules import BATCH_WINDOWS, ENVELOPE, WINDOW, concordance, crossing, differ_pairwise, onset, rank_sign, readings, second_descents

EFFECTIVE_COUNT = 30.8  # the cap row's effective string count (decisions 48: the crossings 311, 699, 1 243 at whole 9)
RUNGS = (0.2, 0.1, 0.05, 0.025)  # the ladder, the same at every size (decisions 53)
PASS_BOUND_SIZE = (192, 9)  # the size whose cap is the corpus's ten-pass bound on the record's corpora, where (a) reads "unread at this corpus"; per cell, `pass_bound` reads it off the corpus
ONSET_WINDOWS_AFTER_CROSSING = 3  # prediction (3)
DESCENT_WINDOWS_AFTER_FOUR_CROSSINGS = 4  # prediction (4)


def train_radius_of(cell):
    """The training radius in grains, from the cell's argv (runs/sweep7.py's usage order)."""
    return int(cell["args"][8])


def size_of(cell):
    return (cell["width"], cell["layers"])


def window_of(step, window=WINDOW):
    """The window convention (decisions 53, third form (1)): the evaluation index of the window holding the step."""
    return math.ceil(step / window)


def copy_reading(cell, name):
    """A copy's readings in the shape rules.readings reads (a plateau-mode cell: "held" and "points"),
    or None when the copy does not exist or was not read."""
    copy = (cell.get("copies") or {}).get(name)
    if copy is None or "points" not in copy:
        return None
    return {"held": copy["held"], "points": copy["points"]}


def pass_bound(cell, effective_count=EFFECTIVE_COUNT, crossings=8, passes=10):
    """Whether this cell's cap is the corpus's pass bound rather than eight crossings (decisions 53,
    second form (ii), third form (6)): ten passes of its corpus in batches of four is fewer steps
    than eight of its crossings."""
    return cell["corpus_windows"] * passes // BATCH_WINDOWS < crossings * crossing(cell["width"], cell["whole"], effective_count)


def has_best(cell):
    return copy_reading(cell, "best") is not None


def onset_check(cell):
    """The onset the harness named against the onset recomputed from the evaluations: they must agree."""
    recomputed = onset(cell["evaluations"], ENVELOPE)
    return {"harness": cell.get("onset_step"), "recomputed": recomputed, "agree": cell.get("onset_step") == recomputed}


# --- Test (1): the ladder per size and radius ----------------------------------------------------


def ladder(cells):
    """`cells`: the cells of one size and radius, any order. The escape step per rung ("none within
    the cap" as None), the concordant and discordant pairs over the pairs with an escape, the rate for
    the next run (the largest rung with no escape within the cap), and the decision: (d) on a
    discordant pair; the ladder extended down one halving when every rung escapes before its onset;
    (a) when no rung has an onset by its cap."""
    by_rung = {cell["lr"]: cell for cell in cells}
    escapes = {lr: by_rung[lr].get("escape_step") for lr in by_rung}
    concordant, discordant = concordance(escapes)
    without = [lr for lr, step in escapes.items() if step is None]
    before_onset = {lr: step is not None and (by_rung[lr].get("onset_step") is None or step <= by_rung[lr]["onset_step"]) for lr, step in escapes.items()}
    any_onset = any(by_rung[lr].get("onset_step") is not None for lr in by_rung)
    at_pass_bound = pass_bound(next(iter(by_rung.values())))
    if discordant:
        decision = "(d): a discordant pair; the escape step is not monotone in the rate at this size (second form (v): at one seed it is a function of the rate by construction, so what fails is monotonicity), and the search needs the seed axis there: decision 5 reconsidered for the rate only, a row for Izzy"
    elif before_onset and all(before_onset.values()):
        decision = "every rung escapes before its onset: the ladder extends down one halving at this size (test (1), kept from the first form)"
    elif not any_onset:
        decision = "(a): no rung leaves the marginal by its cap: " + ("unread at this corpus, a C decision for Izzy (the cap is the pass bound)" if at_pass_bound else "unread at this budget; the next run doubles this size's cap at this radius (R2's move, offline)")
    else:
        decision = "the ladder reads: concordant on every pair with an escape; the next run's rate at this size is the largest rung with no escape within the cap"
    return {"escapes": escapes, "concordant": concordant, "discordant": discordant, "next_rate": max(without) if without else None, "escaped_before_onset": before_onset, "pass_bound": at_pass_bound, "decision": decision}


# --- Test (2): replication by size on the comparable cells ---------------------------------------


def r4_keys(reading_dict):
    """The keys that enter R4 (decisions 53, third form (4)): the slope and the level at each read
    radius past the training radius, six at radius 1 and four at radius 2; the order control is a
    reading of its own and not a key (Buridan, 2026-10-11, departure F1)."""
    return {key: value for key, value in reading_dict.items() if key.startswith("slope_") or key.startswith("level_")}


def null_readings(flat_cells, run_cells, radius):
    """The null per key at a radius: the readings of the flat cells of the record read at the marginal
    (decision 50's four 07c cells, top-level readings) and the marginal copy of every run cell at
    that radius."""
    values = defaultdict(list)
    for cell in flat_cells:
        if train_radius_of(cell) == radius:
            for key, (value, _) in r4_keys(readings(cell, radius)).items():
                values[key].append(value)
    for cell in run_cells:
        if train_radius_of(cell) == radius:
            marginal = copy_reading(cell, "marginal")
            if marginal is not None:
                for key, (value, _) in r4_keys(readings(marginal, radius)).items():
                    values[key].append(value)
    return dict(values)


def best_signs(cell, null, radius):
    """The rank sign of the best copy's reading at every key the null has; None without a best copy."""
    best = copy_reading(cell, "best")
    if best is None:
        return None
    return {key: rank_sign(value, null[key]) for key, (value, _) in r4_keys(readings(best, radius)).items() if key in null}


def _compare(signs_by_size):
    """Per key over the sizes' signs (zeros abstain): None (does not enter), False (agree), True (differ);
    under "_counting_sizes" per key, the number of sizes whose sign is not zero, which tells a key
    that counts at one size only (it abstains and the comparison waits) from a key counting nowhere."""
    keys = sorted({key for signs in signs_by_size for key in signs})
    out = {key: differ_pairwise([signs.get(key, 0) for signs in signs_by_size]) for key in keys}
    out["_counting_sizes"] = {key: sum(1 for signs in signs_by_size if signs.get(key, 0)) for key in keys}
    return out


def replication(run_cells, flat_cells, radius):
    """R4 at one radius: the comparison at each matched rung with three comparable cells (a best copy
    at every size) and at each size's best rung (the largest rung with a best copy); the counting keys
    and whether any differs; the decision (c), the matched rung's verdict governing where the two
    disagree (third form (5))."""
    null = null_readings(flat_cells, run_cells, radius)
    at_radius = [cell for cell in run_cells if train_radius_of(cell) == radius]
    sizes = sorted({size_of(cell) for cell in at_radius})
    by_size = {size: {cell["lr"]: cell for cell in at_radius if size_of(cell) == size} for size in sizes}
    matched = {}
    for lr in RUNGS:
        cells = [by_size[size].get(lr) for size in sizes]
        if len(sizes) == 3 and all(cell is not None and has_best(cell) for cell in cells):
            matched[lr] = _compare([best_signs(cell, null, radius) for cell in cells])
        else:
            matched[lr] = None  # not comparable at this rung
    best_rung = {size: max((lr for lr, cell in by_size[size].items() if has_best(cell)), default=None) for size in sizes}
    if len(sizes) == 3 and all(best_rung[size] is not None for size in sizes):
        at_best = _compare([best_signs(by_size[size][best_rung[size]], null, radius) for size in sizes])
    else:
        at_best = None  # not comparable at this budget: (a) applies to the sizes without an onset

    def verdict(comparison):
        if comparison is None:
            return None
        entering = {key: differs for key, differs in comparison.items() if key != "_counting_sizes" and differs is not None}
        if entering:
            return "differ" if any(entering.values()) else "agree"
        if any(comparison["_counting_sizes"].values()):
            return "every key abstains"  # a key counting at one size only abstains (third form (5)): the comparison waits, no move
        return "no key counts"  # no sign counts at any key at this radius: the N row (decision (c), third branch)

    matched_verdicts = {lr: verdict(comparison) for lr, comparison in matched.items()}
    best_verdict = verdict(at_best)
    governing = [v for v in matched_verdicts.values() if v is not None] or ([best_verdict] if best_verdict else [])
    # several comparable rungs combine as differ over agree over the rest (a choice, marked in the fifth form)
    if not governing:
        decision = "not comparable at this budget: no rung has three comparable cells and no best-rung comparison exists; (a) applies to the sizes without an onset; the comparison waits for the next run"
    elif "differ" in governing:
        decision = "(c): a sign differs across sizes at a counting key: that reading moved with size; the next size is added and the smallest dropped (R4's move, offline)"
    elif "agree" in governing:
        decision = "(c): the signs agree at every counting key: the readings stand for round one (the stop row)"
    elif "every key abstains" in governing:
        decision = "every key abstains (each counts at one size only): the comparison waits for the next run at the same rungs; no move (third form (5))"
    else:
        decision = "(c): no key counts at this radius: the readers do not separate learning from the marginal at this read corpus size; the N row (decision 50 (d)) is written before any sign is spoken, the read corpora doubled as its move"
    return {"null_size": {key: len(v) for key, v in null.items()}, "sizes": sizes, "best_rung": best_rung, "matched": matched, "matched_verdicts": matched_verdicts, "at_best_rung": at_best, "best_rung_verdict": best_verdict, "decision": decision}


# --- Predictions (3) and (4) ---------------------------------------------------------------------


def onset_prediction(run_cells, effective_count=EFFECTIVE_COUNT):
    """Prediction (3): at every rung that learns, the onset lies no later than three windows after the
    crossing at the cell's whole. Per cell with an onset: the crossing, both windows, met or not."""
    out = {}
    for cell in run_cells:
        step = cell.get("onset_step")
        if step is None:
            continue
        cross = crossing(cell["width"], cell["whole"], effective_count)
        out[cell["run"]] = {"size": size_of(cell), "onset_step": step, "crossing": cross, "onset_window": window_of(step), "crossing_window": window_of(cross), "met": window_of(step) <= window_of(cross) + ONSET_WINDOWS_AFTER_CROSSING}
    at_largest = [entry for entry in out.values() if entry["size"] == PASS_BOUND_SIZE]
    return {"cells": out, "unread": not out, "fires": bool(out) and all(entry["met"] for entry in out.values()), "fails_at_largest": bool(at_largest) and not any(entry["met"] for entry in at_largest)}


def second_descent_prediction(run_cells, effective_count=EFFECTIVE_COUNT, width=96):
    """Prediction (4): in at least one radius-2 (96, 7) cell that reaches four crossings, a second
    descent begins within four windows after four crossings (1 244 at whole 9). Every (96, 7) cell
    is reported; the lodged arm is the radius-2 cells."""
    out = {}
    for cell in run_cells:
        if cell["width"] != width:
            continue
        four = 4 * crossing(width, cell["whole"], effective_count)
        reached = cell["train_steps"] >= four
        steps = second_descents(cell["evaluations"], ENVELOPE, after=cell["onset_step"]) if cell.get("onset_step") is not None else []
        in_window = [s for s in steps if window_of(four) < window_of(s) <= window_of(four) + DESCENT_WINDOWS_AFTER_FOUR_CROSSINGS]
        out[cell["run"]] = {"radius": train_radius_of(cell), "four_crossings": four, "reached": reached, "descent_steps": steps, "descent_step": steps[0] if steps else None, "within": bool(in_window)}
    lodged = [entry for entry in out.values() if entry["radius"] == 2 and entry["reached"]]
    return {"cells": out, "radius_2_reaching": len(lodged), "fires": any(entry["within"] for entry in lodged), "unread": not lodged}


# --- The row over the record ---------------------------------------------------------------------


def references_check(run_cells, references):
    """Decisions 54 read beside 53's tests: `references` maps a radius to {"marginal": mean, "floor": mean} from
    the reference logs. Per cell: whether its first evaluation sits at the marginal level (within the envelope),
    and whether its best held loss (at the best copy's step, or the least evaluation after the onset) lies below
    the floor on the same windows, which is 54's Fails-if."""
    out = {}
    for cell in run_cells:
        ref = references.get(train_radius_of(cell))
        if ref is None or not cell["evaluations"]:
            continue
        first = cell["evaluations"][0]
        before_onset = cell["evaluations"][: (cell["onset_step"] // WINDOW - 1)] if cell.get("onset_step") else cell["evaluations"]
        plateau = statistics.median(before_onset) if len(before_onset) >= 2 else first  # the level the cell sits at before its onset; the first evaluation sits above it (53, third form (2))
        best = cell.get("copies", {}).get("best")
        if best is not None and best.get("step"):
            best_loss = cell["evaluations"][best["step"] // WINDOW - 1]
        elif cell.get("onset_step") is not None:
            best_loss = min(cell["evaluations"][cell["onset_step"] // WINDOW - 1 :])
        else:
            best_loss = None
        out[cell["run"]] = {"first": first, "plateau": plateau, "at_marginal": abs(plateau - ref["marginal"]) <= ENVELOPE, "best": best_loss, "below_floor": best_loss is not None and best_loss < ref["floor"], "marginal": ref["marginal"], "floor": ref["floor"]}
    return out


def apply_row(run_cells, flat_cells, radii=(1, 2), references=None):
    """Every test of decisions 53 over the run's cells, with the flat cells of the record for the null, and
    decisions 54's references beside them when given."""
    checks = {cell["run"]: onset_check(cell) for cell in run_cells}
    ladders = {}
    for radius in radii:
        for size in sorted({size_of(c) for c in run_cells}):
            cells = [c for c in run_cells if train_radius_of(c) == radius and size_of(c) == size]
            if cells:
                ladders[(radius, size)] = ladder(cells)
    return {
        "cells": len(run_cells),
        "onset_checks": checks,
        "ladders": ladders,
        "replication": {radius: replication(run_cells, flat_cells, radius) for radius in radii},
        "onset_prediction": onset_prediction(run_cells),
        "second_descent_prediction": second_descent_prediction(run_cells),
        "references": references_check(run_cells, references or {}),
    }


def report(result):
    """The result as text for the log: each test, its numbers, the decision it maps to."""
    lines = [f"decisions 53 over {result['cells']} cells", ""]
    disagree = [name for name, check in result["onset_checks"].items() if not check["agree"]]
    lines.append("onset checks: " + ("the harness's onset agrees with the recomputed one in every cell" if not disagree else "DISAGREE at " + ", ".join(f"{n} ({result['onset_checks'][n]['harness']} vs {result['onset_checks'][n]['recomputed']})" for n in disagree)))
    lines.append("")
    lines.append("test (1), the ladder per size and radius:")
    for (radius, size), lad in sorted(result["ladders"].items()):
        escapes = ", ".join(f"{lr}: {'none within the cap' if step is None else step}" for lr, step in sorted(lad["escapes"].items(), reverse=True))
        lines.append(f"  radius {radius}, {size}: escapes {escapes}; concordant {lad['concordant']}, discordant {lad['discordant']}; next rate {lad['next_rate']}; {lad['decision']}")
    lines.append("")
    lines.append("test (2), replication by size (R4) on the comparable cells:")
    for radius, rep in sorted(result["replication"].items()):
        lines.append(f"  radius {radius}: null sizes per key {rep['null_size']}; keys at this radius {len(rep['null_size'])} (second form (iv): the count of counting keys is read against it); best rung per size {rep['best_rung']}")
        for lr, comparison in sorted(rep["matched"].items(), reverse=True):
            if comparison is None:
                lines.append(f"    matched rung {lr}: not comparable")
            else:
                entering = {k: v for k, v in comparison.items() if k != "_counting_sizes" and v is not None}
                one_size = [k for k, n in comparison["_counting_sizes"].items() if n == 1]
                lines.append(f"    matched rung {lr}: {rep['matched_verdicts'][lr]} ({len(entering)} of {len(rep['null_size'])} keys count: {', '.join(f'{k} {v}' for k, v in entering.items()) or 'none'}{'; counting at one size only, abstaining: ' + ', '.join(one_size) if one_size else ''})")
        lines.append("    best rungs: " + ("not comparable at this budget" if rep["at_best_rung"] is None else f"{rep['best_rung_verdict']} ({', '.join(f'{k} {v}' for k, v in rep['at_best_rung'].items() if k != '_counting_sizes' and v is not None) or 'no counting key'})"))
        lines.append(f"    {rep['decision']}")
    lines.append("")
    pred3 = result["onset_prediction"]
    lines.append(f"prediction (3), the onset no later than {ONSET_WINDOWS_AFTER_CROSSING} windows after the crossing: " + ("unread (no cell learned)" if pred3["unread"] else ("fires" if pred3["fires"] else "fails")) + "; maps into decision 48's next form (decision (e))")
    for name, entry in sorted(pred3["cells"].items()):
        lines.append(f"  {name}: onset {entry['onset_step']} (window {entry['onset_window']}), crossing {entry['crossing']} (window {entry['crossing_window']}): {'met' if entry['met'] else 'NOT met'}")
    if pred3["fails_at_largest"]:
        lines.append(f"  the Fails-if arm: at {PASS_BOUND_SIZE} the onset lies more than {ONSET_WINDOWS_AFTER_CROSSING} windows after the crossing at every rung that learns; the crossing is not the onset's clock at the third size")
    lines.append("")
    pred4 = result["second_descent_prediction"]
    lines.append(f"prediction (4), a second descent within {DESCENT_WINDOWS_AFTER_FOUR_CROSSINGS} windows after four crossings in a radius-2 (96, 7) cell: " + ("unread (no radius-2 cell reached four crossings)" if pred4["unread"] else ("fires" if pred4["fires"] else "fails: the cap's unit loses one of its two supports (second form (iii)); the other, the onsets selecting the 39.6-bit count (48), stands or falls with prediction (3)")) + "; maps into decision 48's next form (decision (e))")
    for name, entry in sorted(pred4["cells"].items()):
        lines.append(f"  {name} (radius {entry['radius']}): four crossings {entry['four_crossings']}, {'reached' if entry['reached'] else 'not reached'}; descents at {entry['descent_steps'] or 'none'}; {'one within' if entry['within'] else 'none within'} windows {window_of(entry['four_crossings']) + 1} to {window_of(entry['four_crossings']) + DESCENT_WINDOWS_AFTER_FOUR_CROSSINGS}")
    refs = result.get("references") or {}
    lines.append("")
    if refs:
        at = sum(1 for e in refs.values() if e["at_marginal"])
        below = [n for n, e in refs.items() if e["below_floor"]]
        lines.append(f"decisions 54 beside the tests: {at} of {len(refs)} cells' plateaus before the onset sit at the marginal level (within the envelope; the first evaluation sits above the plateau, 53 third form (2)); " + (f"FAILS-IF: best held loss below the floor at {', '.join(below)}" if below else "no cell's best held loss lies below the floor"))
        for name, e in sorted(refs.items()):
            lines.append(f"  {name}: plateau {e['plateau']:.4f} (first {e['first']:.4f}) against marginal {e['marginal']:.4f} ({'at' if e['at_marginal'] else 'off'}); best {e['best'] if e['best'] is None else round(e['best'], 4)} against floor {e['floor']:.4f}")
    else:
        lines.append("decisions 54 beside the tests: no reference logs given")
    return "\n".join(lines)
