# Variance 8 (2026-10-29): the floor's own variance at a particle count, read as the pilot's file asked
# before the row ("the particle count for the floor is measured against the floor's own variance (two seeds
# of the filter on the same windows)"): reference8.py run on one cell and copy with the same windows and
# particles under different particle seeds (runs/launch-ref8-var.sh), the floor and S per seed side by side
# at every reading point and every rank the record's rule allows (every window reaches), with the span
# (max minus min over the seeds) that any difference read between cells at that point must exceed to be a
# reading and not the filter's noise. A point whose floor is exact (distance 0) must span 0.000: the check
# that the seeds differ in the filter alone. Reads the logs given; prints to stdout, numbers to the log
# only (the rule). Usage: variance8.py <reference8 logs of one or more cells, any seeds...>
import json
import sys
from collections import defaultdict

groups = defaultdict(dict)
for path in sys.argv[1:]:
    with open(path) as f:
        text = f.read().strip()
    if not text:
        print(f"{path}: empty (the run failed or is running; see its .err)")
        continue
    d = json.loads(text)
    key = (d["run"], d["copy"], d["windows"], d["particles"])
    seed = d.get("particle_seed", 11)
    if seed in groups[key]:
        print(f"{path}: seed {seed} twice for {key}; the first is kept")
        continue
    groups[key][seed] = d


def span(xs):
    return max(xs) - min(xs)


for (run, copy, windows, particles), by_seed in sorted(groups.items()):
    seeds = sorted(by_seed)
    print(f"{run} copy {copy}: {windows} windows, {particles} particles, seeds {seeds}")
    if len(seeds) < 2:
        print("  one seed only: no span to read")
        continue
    first = by_seed[seeds[0]]
    for r, p0 in first["points"].items():
        ps = [by_seed[s]["points"].get(r) for s in seeds]
        if any(p is None for p in ps):
            print(f"  read {r:>4}: missing in a seed; skipped")
            continue
        same = all(p["windows"] == p0["windows"] and abs(p["truth_distance"] - p0["truth_distance"]) < 1e-12 for p in ps)
        ranks = min(p["whole"] for p in ps)
        est = {str(p["floor_estimate"]) for p in ps}
        print(f"  read {r:>4}: truth d {p0['truth_distance']:.3f}  ranks read {ranks}  floor {'/'.join(sorted(est))}  windows {'same' if same else 'DIFFER'}")
        for k in range(ranks):
            floors = [p["floor_by_rank"][k] for p in ps]
            ss = [p["S_by_rank"][k] for p in ps]
            models = [p["model_by_rank"][k] for p in ps]
            m_note = "" if span(models) < 1e-9 else f"  MODEL DIFFERS by {span(models):.4f}"
            print(f"    rank {k + 1}: floor " + " ".join(f"{x:.4f}" for x in floors) + f"  span {span(floors):.4f}  mean {sum(floors) / len(floors):.4f}"
                  + "   S " + " ".join(f"{x:.3f}" for x in ss) + f"  span {span(ss):.3f}" + m_note)
