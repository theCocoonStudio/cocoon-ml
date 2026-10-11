# Decisions 53 applied to the record of a run: loads the JSON lines of the run's cells (runs/sweep7.py
# in mode "cap") and the four 07c cells for the null, applies every test of the row
# (cocoonml/run_row.py) and prints the report: which decision each outcome maps to, with the numbers
# it rests on. Nothing here is a reading of a curve; the readers' numbers stay in the logs.
# Usage: row53.py <log_dir> [prefix=sweep-07e] [flat_prefix=sweep-07c]
import glob
import os
import sys

from cocoonml.rules import parse_cell
from cocoonml.run_row import apply_row, report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: row53.py <log_dir> [prefix=sweep-07e] [flat_prefix=sweep-07c]")
    log_dir = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "sweep-07e"
    flat_prefix = sys.argv[3] if len(sys.argv) > 3 else "sweep-07c"

    def load(pattern):
        cells = []
        for path in sorted(glob.glob(os.path.join(log_dir, pattern))):
            with open(path) as f:
                cell = parse_cell(f.read())
            if cell is not None:
                cells.append(cell)
        return cells

    run_cells = load(f"{prefix}-r*-w*-lr*.log")
    flat = [c for c in load(f"{flat_prefix}-r*-w*.log") if c["width"] in (144, 192)]  # decision 50's four flat cells; the (96, 7) cells learned
    missing = len(glob.glob(os.path.join(log_dir, f"{prefix}-r*-w*-lr*.log"))) - len(run_cells)
    print(f"{len(run_cells)} cells with a JSON line, {missing} logs without one (still running or failed); {len(flat)} flat cells for the null")
    print()
    print(report(apply_row(run_cells, flat)))
