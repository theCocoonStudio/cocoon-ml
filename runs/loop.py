# The self-adjusting runner of docs/rules.md (2026-10-09): runs sweep 07 cells at a parameter vector
# over three sizes, reads their JSON lines, applies the first row that holds (cocoonml/rules.py holds
# the rows and nothing else), appends one entry to the loop log, and repeats; stops when no row fires,
# when the next size's estimated time exceeds the host, or at the iteration limit. Cells run as
# runs/README.md says: from the archive, PYTHONPATH at it, OMP_NUM_THREADS=2, one process each, all
# of an iteration in parallel, stdout to <log_dir>/<name>.log. A log whose JSON line was produced at
# the current vector is read, not re-run (resume); a log of the same name from a moved vector is set
# aside under a stamp so its numbers stay. The loop itself imports cocoonml.rules, so run it with
# PYTHONPATH at the archive too. Numbers stay in the cell logs (the rule); the entry carries none.
# Usage: loop.py <archive_dir> <log_dir> <width> <layers> <length> <corpus> <read_size> <cap> <lr: one rate, or one per size a,b,c> <train_radii: a,b> [max_iterations=6] [loop_log=runs/loop-log.md] [prefix=sweep-07]
# The rate is per size (decisions 40): R1 names the cells that fired and the loop halves the rate of each of their sizes and re-runs those cells only.
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime

from cocoonml.rules import Ensemble, Vector, apply_rows, cell_at, cell_command, cell_name, host_exceeds, log_entry, next_size, parse_cell, sizes, sizes_that_fired_r1

# The ensemble is not a parameter of the loop (docs/rules.md: the antecedent of the predictions,
# moved by no row); these are sweep 07's values from runs/README.md. The training radii come from
# the command line because they name the cells, not because a row may move them.
FORMS = 4
NONTERMINALS = 3
RESOLUTION = 4
TABLES = 2
SEED = 1
READ_RADII = ("0", "1", "2", "3", "free")
STEPS_MEAN = 1.0
SWITCH_MEAN = 0.5
PATIENCE = 3
MINIMUM = 300
LEAN = "3/4"

if __name__ == "__main__":
    if len(sys.argv) < 11:
        sys.exit("usage: loop.py <archive_dir> <log_dir> <width> <layers> <length> <corpus> <read_size> <cap> <lr: one, or one per size a,b,c> <train_radii: a,b> [max_iterations=6] [loop_log=runs/loop-log.md] [prefix=sweep-07]")
    archive_dir, log_dir = sys.argv[1], sys.argv[2]
    width, layers, length, corpus, read_size, cap = (int(x) for x in sys.argv[3:9])
    given_rates = [float(x) for x in sys.argv[9].split(",")]
    lr = given_rates[0]
    train_radii = tuple(int(r) for r in sys.argv[10].split(","))
    max_iterations = int(sys.argv[11]) if len(sys.argv) > 11 else 6
    loop_log = sys.argv[12] if len(sys.argv) > 12 else os.path.join("runs", "loop-log.md")
    prefix = sys.argv[13] if len(sys.argv) > 13 else "sweep-07"

    ensemble = Ensemble(FORMS, NONTERMINALS, RESOLUTION, TABLES, SEED, train_radii, READ_RADII, STEPS_MEAN, SWITCH_MEAN, PATIENCE, MINIMUM, LEAN)
    vector = Vector(width, layers, length, corpus, read_size, cap, lr)
    current = sizes(width, layers)
    rates = {size: given_rates[i] for i, size in enumerate(current) if i < len(given_rates)}  # the rate per size; a size not named takes the vector's
    script = os.path.join(archive_dir, "runs", "sweep7.py")
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONPATH=archive_dir)
    os.makedirs(log_dir, exist_ok=True)

    def write(entry):
        with open(loop_log, "a") as f:
            f.write(entry + "\n")
        print(entry, flush=True)

    for iteration in range(max_iterations):
        cells, running = {}, []
        for train_radius in train_radii:
            for w, l in current:
                name = cell_name(train_radius, w, prefix)
                at_size = replace(vector, width=w, layers=l)
                argv = cell_command(sys.executable, script, name, ensemble, at_size, train_radius, rates)
                path = os.path.join(log_dir, name + ".log")
                existing = None
                if os.path.exists(path):
                    with open(path) as f:
                        existing = parse_cell(f.read())
                if existing is not None and cell_at(existing, at_size, train_radius, rates):
                    cells[(train_radius, w)] = existing  # resume: this cell ran at this vector
                    continue
                if existing is not None:  # the same name at a moved vector: keep its numbers under a stamp
                    stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d-%H%M%S")
                    os.replace(path, os.path.join(log_dir, f"{name}.{stamp}.log"))
                out = open(path, "w")
                running.append(((train_radius, w), path, out, subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)))
        for key, path, out, process in running:
            process.wait()
            out.close()
            with open(path) as f:
                cells[key] = parse_cell(f.read())
        failed = [cell_name(key[0], key[1], prefix) for key, c in sorted(cells.items()) if c is None]
        if failed:
            sys.exit(f"no JSON line from {', '.join(failed)}: see {log_dir}")

        mean_artifacts_per_window = sum(c["corpus_artifacts"] for c in cells.values()) / sum(c["corpus_windows"] for c in cells.values())
        row, following, detail = apply_rows(cells, train_radii, vector, mean_artifacts_per_window)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if row == "stop":
            write(log_entry(stamp, vector, current, cells, row, detail, None))
            break
        if row == "R1":
            for size in sizes_that_fired_r1(cells):
                rates[size] = rates.get(size, vector.lr) / 2  # that size's rate halves; its cells re-run (the others resume by cell_at)
            coming = current
        elif row == "R4":
            coming = [current[1], current[2], next_size(*current[2])]
            following = replace(vector, width=coming[0][0], layers=coming[0][1])
        else:
            coming = current
        largest = current[-1]
        last_seconds = max(cells[(train_radius, largest[0])]["seconds"] for train_radius in train_radii)
        exceeds, host_detail = host_exceeds(last_seconds, largest[0], largest[1], coming[-1][0], coming[-1][1])
        if exceeds:
            write(log_entry(stamp, vector, current, cells, "host", f"{row} still fires ({detail}); {host_detail}", None))
            break
        write(log_entry(stamp, vector, current, cells, row, detail, following))
        vector, current = following, coming
    else:
        print(f"iteration limit {max_iterations} reached; the vector stands at {vector} over sizes {current}", flush=True)
