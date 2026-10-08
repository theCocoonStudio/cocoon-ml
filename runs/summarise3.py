# Reads a sweep-03 log and prints the numbers compactly (for Claude's round one; numbers stay here).
import json, sys
rows = None
for line in open(sys.argv[1]):
    if line.startswith('{"run"'):
        rows = json.loads(line)
if rows is None:
    print("not finished"); sys.exit(0)
print(rows["run"], rows["args"], f'{rows["seconds"]:.0f}s')
def f(x, w=6):
    return "   -  " if x is None else f"{x:{w}.3f}"
for r in rows["rows"]:
    print(f'table {r["table"]} model {r["model"]}: train {r["train_first"]:.2f} -> {r["train_mid"]:.2f} -> {r["train_last"]:.2f} (meaning floor {r["train_floor_meaning"]:.3f}); probe r2 {f(r["probe_r2_heldout"])} on {r["probe_windows"]} windows')
    for d, v in r["per_depth"].items():
        fe = " ".join(f"{x:5.2f}" for x in v["form_excess_by_rank"]) or "  (none)"
        me = " ".join(f"{x:5.2f}" for x in v["meaning_excess_by_rank"]) or "  (none)"
        print(f'   d={d:>2} dmag {v["delta_magnitude"]:.3f} floors f {v["floors"]["form"]:.3f} m {v["floors"]["meaning"]:.3f} | ident {f(v["identification_error"])} cov {f(v["coverage"])} shown {f(v["window_drift_shown"])} | neff {f(v["n_eff_moved"])} foreign {f(v["foreignness"])} | art/win {v["artifacts_per_window"]:.2f} | form xs [{fe}] | meaning xs [{me}]')
