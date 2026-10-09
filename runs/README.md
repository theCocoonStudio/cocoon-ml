# runs/

The scripts that produced the numbers each file in `docs/reviews/` argues against. None of them is part of the package; they are the record of how a sweep was run, so that a round can be checked and a cell re-run from the repo. The test suite compiles them (`tests/test_runs.py`).

How a sweep runs: from a `git archive` of the branch head unpacked into a scratch directory, with `PYTHONPATH` pointing at it, so checkouts and restarts do not change the code under a running cell; one cell per process; each cell writes its progress and then one JSON line (`{"run": ...}`) to its log; the summariser for that sweep merges the logs. Numbers stay in the logs under the rule stated on the concept page: two dialectical rounds and two replications before any number is spoken.

| sweep | script | launched as (`$t`, `$m` = table seed, model seed) | round one |
| --- | --- | --- | --- |
| smoke 01, 02 | `smoke-01.py`, `smoke-02.py` | `python3 runs/smoke-01.py` (arguments fixed in the file) | `2026-10-08-count-round-1.md` |
| 01 | `sweep-01.py` | `python3 runs/sweep-01.py` (arguments fixed in the file) | `2026-10-08-sweep-01-round-1.md` |
| 02a, 02b, 02c | `sweep2.py` | per its usage line: a the base size, b the larger table and context, c the wider model | `2026-10-08-sweep-02-round-1.md` |
| 03a | `sweep3.py` | `python3 runs/sweep3.py sweep-03a 4 3 8 4 14 400 100 1 2` | `2026-10-08-sweep-03-round-1.md` |
| 03b | `sweep3.py` | `python3 runs/sweep3.py sweep-03b 6 4 8 4 20 400 100 1 2` | same |
| 03c | `sweep3.py` | `python3 runs/sweep3.py sweep-03c 4 3 8 6 14 400 100 1 2` | same |
| 03d | `sweep3.py` | `python3 runs/sweep3.py sweep-03d 4 3 8 4 14 400 100 2 2` | same |
| 04a | `sweep4.py` | `python3 runs/sweep4.py sweep-04a 4 3 4 4 40 100 64 0,1,2 0 $t $m 1500`, t and m in 1..3 | `2026-10-08-sweep-04-round-1.md` |
| 04b | `sweep4.py` | `python3 runs/sweep4.py sweep-04b 6 4 4 4 48 100 64 0,1,2 0 $t $m 1500`, t and m in 1..2 | same |
| 05a | `sweep5.py` | `python3 runs/sweep5.py sweep-05a 4 3 4 4 40 100 64 0,1,2 0 $t $m 1500 2`, t and m in 1..3 | `2026-10-08-sweep-05-round-1.md` |
| 05b | `sweep5.py` | `python3 runs/sweep5.py sweep-05b 4 3 4 4 40 100 64 0,1,2 0 $t $m 3000 2 0.1 3 300`, t and m in 1..3; launched 2026-10-08 14:46 | pending |
| 06 | `sweep6.py` | as 05 plus `[train_radius_grains] [steps_per_context]`: `python3 runs/sweep6.py <name> ... 3000 2 0.1 3 300 <r_train> <steps>`; smoke `smoke-06 3 2 4 3 12 6 8 0,1 0 1 1 60 2 0.3 3 10 1 1` run 2026-10-08 | pending: the training radius and the steps per context are Izzy's numbers |
| loop | `loop.py` | the self-adjusting runner of `docs/rules.md` (its rows in `cocoonml/rules.py`, tested): `PYTHONPATH=<archive> python3 runs/loop.py <archive> ~/.claude/scratch/runs 96 7 64 42000 2000 6000 0.025,0.0125,0.0125 1,2 6 runs/loop-log.md sweep-07b`, resuming the cells already run at the vector; one entry per iteration in `runs/loop-log.md`, no number in it | the log is the record of what moved and why |
| 07 | `sweep7.py` | the complete system over the corpus layer on the array model (`cocoonml/array_attention.py`, the derived initialisation scale); smoke `PYTHONPATH=. python3 runs/sweep7.py smoke-07 3 2 4 3 2 16 2 1 1 0.5 40 0,2,free 20 30 1 0.3 3 5` run 2026-10-09 on both models; the derived-size cells: `OMP_NUM_THREADS=2 python3 runs/sweep7.py sweep-07-r$r-w$w 4 3 4 $w $l 64 2 $r 1 0.5 42000 0,1,2,3,free 2000 6000 1 0.025 3 300` with (w, l, lr) in (96, 7, 0.025), (144, 8, 0.0125), (192, 9, 0.0125) and r in 1, 2, the lr in place of 0.025 (decisions 33, 34, 40; the rate per size by rule R1 of `docs/rules.md`, halved from 0.1 until four hundred steps run finite); launched 2026-10-09 14:58 as sweep-07b from the archive of 0803594 | pending: launched 2026-10-09 afternoon |

The smoke lines of sweeps 03 to 05 (`smoke-03`, `smoke-04`, `smoke-04r`, `smoke-05`) are the same scripts at toy size, run once before each launch. The summarisers (`summarise3.py`, `summarise4.py`, `summarise5.py`) take the sweep name and read `runs/<name>-t*m*.log` beside them (`summarise3.py` takes one log).
