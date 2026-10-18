# Summarise 8 (2026-10-18): the unresolved share S read by reference8.py across cells, one table: rows a
# cell and a copy, columns the reading radii, each entry S at the ranks the record's rule allows (every
# window reaches) and, in brackets, the deep curve's next ranks with their position counts, so the k
# axis past the cut is visible with its selection. Reads the logs given; prints to stdout, numbers to
# the log only (the rule).
# Usage: summarise8.py <reference8 logs...>
import json
import sys

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        text = f.read().strip()
    if not text:
        rows.append((path, None))
        continue
    rows.append((path, json.loads(text)))


def fmt(xs, n=4):
    return " ".join("  -  " if x is None else f"{x:5.2f}" for x in xs[:n])


for path, d in rows:
    if d is None:
        print(f"{path}: empty (the run failed; see its .err)")
        continue
    print(f"{d['run']} copy {d['copy']} (step {d['copy_step']}, held {d['copy_held_loss']:.3f}); training radius {d['train_radius_grains']} grains; ball {d['ball_states']} states; {d['windows']} windows a reading; {d['seconds']} s")
    for r, p in d["points"].items():
        skip = f" floor on {p['floor_windows']} (retries {p['floor_retries']}, skipped {p['floor_skipped']})" if p.get("floor_skipped") or p.get("floor_retries") else ""
        print(f"  read {r:>4}: truth d {p['truth_distance']:.3f}  whole {p['whole']}  S by rank [{fmt(p['S_by_rank'], 6)}]  deep ranks 1..6 [{fmt(p['deep']['S_by_rank'], 6)}] counts {p['deep']['counts'][:6]}{skip}")
        print(f"             model [{fmt(p['model_by_rank'], 6)}] floor [{fmt(p['floor_by_rank'], 6)}] marginal [{fmt(p['marginal_read_by_rank'], 6)}]")
