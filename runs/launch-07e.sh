#!/bin/bash
# sweep-07e, the run of decisions 53 (the run's row at the statistical layer, 2026-10-10): twenty-four cells,
# three sizes x two training radii x the ladder of rates 0.2, 0.1, 0.05, 0.025; one seed; each cell to its cap
# (2 488, 5 592 and 9 127 steps: eight crossings at the two smaller sizes, the ten-pass bound at the largest) or
# its escape, no stopping rule; three copies kept, the marginal and the best read (runs/sweep7.py in mode "cap").
# Ensemble and seed as 07c and 07d (runs/README.md); the radius-1 cells at 0.05 and 0.025 reproduce the record's
# c, a2, b and b2 by the seed and extend them to the cap. Largest cells first: the (192, 9) cells are the clock.
# Runs from a git archive of the branch head so checkouts and restarts do not change the code under a cell.
# Usage: launch-07e.sh <archive_dir> <log_dir> [parallel=6] [--dry-run]
#   e.g. nohup bash runs/launch-07e.sh ~/.claude/scratch/ml-run14 ~/.claude/scratch/runs 6 > ~/.claude/scratch/runs/launch-07e.out 2>&1 &
set -u
ARCHIVE=${1:?archive_dir}
LOGS=${2:?log_dir}
PARALLEL=${3:-6}
DRY=${4:-}
export ARCHIVE LOGS

cells() {
  for size in "192 9 9127" "144 8 5592" "96 7 2488"; do
    set -- $size
    for r in 1 2; do
      for lr in 0.2 0.1 0.05 0.025; do
        echo "sweep-07e-r$r-w$1-lr$lr $1 $2 $3 $r $lr"
      done
    done
  done
}

run_cell() { # name width layers cap train_radius lr
  echo "$(date '+%Y-%m-%d %H:%M:%S') start $1"
  OMP_NUM_THREADS=2 PYTHONPATH="$ARCHIVE" python3 "$ARCHIVE/runs/sweep7.py" "$1" 4 3 4 "$2" "$3" 64 2 "$5" 1 0.5 42000 0,1,2,3,free 2000 "$4" 1 "$6" 3 300 3/4 cap "$LOGS/copies" > "$LOGS/$1.log" 2> "$LOGS/$1.err" < /dev/null
  echo "$(date '+%Y-%m-%d %H:%M:%S') end $1 exit $?"
}
export -f run_cell

if [ "$DRY" = "--dry-run" ]; then
  cells | while read -r name w l cap r lr; do
    echo "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/sweep7.py $name 4 3 4 $w $l 64 2 $r 1 0.5 42000 0,1,2,3,free 2000 $cap 1 $lr 3 300 3/4 cap $LOGS/copies > $LOGS/$name.log 2> $LOGS/$name.err"
  done
  exit 0
fi

mkdir -p "$LOGS/copies"
echo "$(date '+%Y-%m-%d %H:%M:%S') launch-07e: $(cells | wc -l) cells, $PARALLEL at a time, two threads each, from $ARCHIVE"
cells | xargs -P "$PARALLEL" -L 1 bash -c 'run_cell "$@"' _
echo "$(date '+%Y-%m-%d %H:%M:%S') launch-07e: all cells ended"
