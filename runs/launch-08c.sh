#!/bin/bash
# sweep-08c, the training-drift pilot's second run (2026-10-19, the pilot's follow-up: "a longer window
# and more particles"): the 08b cells again at window length 128 instead of 64, so every window holds
# about four whole artifacts and the k axis has four ranks instead of two; then reference8 on the best
# and marginal copies of both cells at 64 windows with 1 000 particles instead of 200, the far readings
# included, so the floor at a distance is read with five times the particles. Two processes at a time,
# never more (the box held seven on 2026-10-18 and the harness stopped them); free is read before each
# launch; one process's resident memory was measured on a smoke of the same shape before this file ran.
# Runs from a git archive of main, as the other launches do.
# Usage: launch-08c.sh <archive_dir> <log_dir> [--dry-run]
set -u
ARCHIVE=${1:?archive_dir}
LOGS=${2:?log_dir}
DRY=${3:-}
cells() {
  echo "sweep-08c-s1-r3-w96-lr0.05 3"
  echo "sweep-08c-s1-r6-w96-lr0.05 6"
}
train_for() { # name train_radius
  echo "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/sweep8.py $1 4 8 96 7 128 2 $2 1 0.5 42000 0,1,3,6,free 2000 2488 1 0.05 3 300 1.0 1.0 cap $LOGS/copies 200 > $LOGS/$1.log 2> $LOGS/$1.err"
}
read_for() { # name copy
  echo "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/reference8.py $LOGS/$1.log $2 64 1000 0,1,3,6,free > $LOGS/reference8-$1-$2.log 2> $LOGS/reference8-$1-$2.err"
}
if [ "$DRY" = "--dry-run" ]; then
  while read -r name radius; do train_for "$name" "$radius"; read_for "$name" best; read_for "$name" marginal; done <<< "$(cells)"
  exit 0
fi
mkdir -p "$LOGS/copies"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
avail() { awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
echo "$(stamp) launch-08c: the two training cells, two at a time; free $(avail) MB available"
while read -r name radius; do
  bash -c "$(train_for "$name" "$radius") < /dev/null" &
done <<< "$(cells)"
wait
echo "$(stamp) launch-08c: both cells ended; the references, one cell's two copies at a time"
while read -r name _; do
  echo "$(stamp) start $name best+marginal; free $(avail) MB available"
  bash -c "$(read_for "$name" best) < /dev/null" &
  bash -c "$(read_for "$name" marginal) < /dev/null" &
  wait
done <<< "$(cells)"
echo "$(stamp) launch-08c: the four references ended"
