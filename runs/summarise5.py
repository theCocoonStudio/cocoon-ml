# Merges the one-line-per-cell logs of a sweep-05 run and prints them compactly (Claude's reading; numbers stay here).
import glob, json, sys
name = sys.argv[1]
cells = []
for f in sorted(glob.glob(f"{name}-t*m*.log")):
    lines = [l for l in open(f) if l.startswith('{"run"')]
    if lines:
        cells.append(json.loads(lines[-1]))
print(f"{name}: {len(cells)} cells finished")
def f(x, w=6):
    return "   -  " if x is None else f"{x:{w}.3f}"
def curve(xs):
    return " ".join(f"{x:5.2f}" for x in xs) or " (none)"
for c in cells:
    ev = " ".join(f"{e:.2f}" for e in c["evaluations"][-3:])
    print(f'table {c["table"]} model {c["model"]} layers {c["layers"]}: {c["seconds"]:.0f}s, {c["train_steps"]} steps{" (cap)" if c["hit_cap"] else ""}, held-out last [{ev}]; probe r2 {f(c["probe_r2_heldout"])} on {c["probe_windows"]}')
    s = c["same"]
    print(f'   same    floors f {s["floors"]["form"]:.3f} m {s["floors"]["meaning"]:.3f} | ident {f(s["identification_error"])} excess {f(s["identification_excess"])} cov {f(s["coverage"])} | art/win {s["artifacts_per_window"]:.2f}')
    print(f'           form xs [{curve(s["form_excess_by_rank"])}] | meaning xs [{curve(s["meaning_excess_by_rank"])}] | scramble cost [{curve(s["scramble_cost_by_rank"])}]')
    for r, v in c["points"].items():
        co = v["coordinates"]
        print(f'   radius {r} distance {co["distance"]:.3f} entropy f {co["entropy_form"]:.3f} inc {co["incidence"]:.2f} | ident {f(v["identification_error"])} excess {f(v["identification_excess"])} shown {f(v["window_drift_shown"])} | art/win {v["artifacts_per_window"]:.2f}')
        print(f'           form minus same [{curve(v["form_recovery_vs_same"])}] | meaning xs [{curve(v["meaning_excess_by_rank"])}] minus same [{curve(v["meaning_recovery_vs_same"])}] | scramble cost [{curve(v["scramble_cost_by_rank"])}]')
