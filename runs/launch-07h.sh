#!/bin/bash
# sweep-07h, the one-table control (decisions 56; docs/literature.md entry 9, Makkuva et al.): one
# graded table producing in place (tables 1, steps 0), (96, 7) at 0.05 to the cap 2 488 in mode cap,
# the 07e radius-1 cell's other arguments; the plateau and the drop where the bigram is in-weights by
# construction, for the onsets of 07e and 08 to be read against. Runs from a git archive of the branch
# head, as the other launches do.
# Usage: launch-07h.sh <archive_dir> <log_dir> [--dry-run] [rate=0.05]  (the rerun at 0.025 after an escape, decisions 56 Fails-if)
set -u
ARCHIVE=${1:?archive_dir}
LOGS=${2:?log_dir}
DRY=${3:-}
RATE=${4:-0.05}
NAME=sweep-07h-t1-w96-lr$RATE
command_for() {
  echo "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/sweep7.py $NAME 4 3 4 96 7 64 1 1 0 0.5 42000 0,1,2,3,free 2000 2488 1 $RATE 3 300 3/4 cap $LOGS/copies > $LOGS/$NAME.log 2> $LOGS/$NAME.err"
}
if [ "$DRY" = "--dry-run" ]; then
  command_for
  exit 0
fi
mkdir -p "$LOGS/copies"
echo "$(date '+%Y-%m-%d %H:%M:%S') start $NAME"
bash -c "$(command_for) < /dev/null"
echo "$(date '+%Y-%m-%d %H:%M:%S') launch-07h: the cell ended"
