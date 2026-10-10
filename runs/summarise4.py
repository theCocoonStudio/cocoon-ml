# Merges the one-line-per-cell logs of a sweep-04 run and prints them compactly (Claude's reading; numbers stay here).
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
    print(f'table {c["table"]} model {c["model"]}: {c["seconds"]:.0f}s, train {c["train_steps"]} steps{" (cap)" if c["hit_cap"] else ""} {c["train_first"]:.2f} -> {c["train_mid"]:.2f} -> {c["train_last"]:.2f}; probe r2 {f(c["probe_r2_heldout"])} on {c["probe_windows"]}')
    s = c["same"]
    print(f'   same      floors f {s["floors"]["form"]:.3f} m {s["floors"]["meaning"]:.3f} | ident {f(s["identification_error"])} cov {f(s["coverage"])} | neff {f(s["n_eff"])} | art/win {s["artifacts_per_window"]:.2f} | form xs [{curve(s["form_excess_by_rank"])}] | meaning xs [{curve(s["meaning_excess_by_rank"])}]')
    for r, v in c["per_radius"].items():
        print(f'   radius {r} dmag {v["delta_magnitude"]:.3f} inc {v["incidence_delta"]:.2f} floors f {v["floors"]["form"]:.3f} m {v["floors"]["meaning"]:.3f} | ident {f(v["identification_error"])} cov {f(v["coverage"])} shown {f(v["window_drift_shown"])} | neff {f(v["n_eff"])} foreign {f(v["foreignness"])} | art/win {v["artifacts_per_window"]:.2f}')
        print(f'            form xs [{curve(v["form_excess_by_rank"])}] minus same [{curve(v["form_recovery_vs_same"])}]')
        print(f'            meaning xs [{curve(v["meaning_excess_by_rank"])}] minus same [{curve(v["meaning_recovery_vs_same"])}]')
