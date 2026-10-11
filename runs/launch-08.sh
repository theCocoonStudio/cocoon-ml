#!/bin/bash
# sweep-08, the Markov-chain world's first cells (decisions 55, "The first cells"): their stationary
# case (steps 0, the floor exact) and the first drift cell (steps 1, training radius 1), both (96, 7)
# at 0.05 to the cap 2 488 in mode cap, four states at resolution 8, two chains, alpha 1, end 1.
# Runs from a git archive of the branch head, as the other launches do; after 07e frees the box.
# Usage: launch-08.sh <archive_dir> <log_dir> [--dry-run]
set -u
ARCHIVE=${1:?archive_dir}
LOGS=${2:?log_dir}
DRY=${3:-}
cells() {
  echo "sweep-08-s0-w96-lr0.05 0"
  echo "sweep-08-s1-r1-w96-lr0.05 1"
}
command_for() { # name steps
  echo "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/sweep8.py $1 4 8 96 7 64 2 1 $2 0.5 42000 0,1,2,3,free 2000 2488 1 0.05 3 300 1.0 1.0 cap $LOGS/copies 200 > $LOGS/$1.log 2> $LOGS/$1.err"
}
if [ "$DRY" = "--dry-run" ]; then
  cells | while read -r name steps; do command_for "$name" "$steps"; done
  exit 0
fi
mkdir -p "$LOGS/copies"
cells | while read -r name steps; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') start $name"
  bash -c "$(command_for "$name" "$steps") < /dev/null" &
done
wait
echo "$(date '+%Y-%m-%d %H:%M:%S') launch-08: both cells ended"
