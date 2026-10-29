#!/bin/bash
# The floor's own variance (2026-10-19, the second pilot's reading: "the floor's variance measured by two filter
# seeds on the same windows" before the row): reference8 on the best copy of the three drift cells of the record
# (08 r1, 08b r3, 08b r6; window 64) at 200 particles, 64 windows, the reading radii as the first reading had
# them, with particle seeds 12 and 13; the seed-11 runs are the first reading's logs. Two processes at a time,
# free read before each pair, from an archive of the branch head. Usage: launch-ref8-var.sh <archive> <log_dir>
set -u
ARCHIVE=${1:?archive_dir}
LOGS=${2:?log_dir}
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
avail() { awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
runs() {
  echo "sweep-08-s1-r1-w96-lr0.05 0,1,2,3,free"
  echo "sweep-08b-s1-r3-w96-lr0.05 0,1,3,6,free"
  echo "sweep-08b-s1-r6-w96-lr0.05 0,1,3,6,free"
}
echo "$(stamp) launch-ref8-var: three cells, two seeds each, two processes at a time; free $(avail) MB available"
while read -r name radii; do
  echo "$(stamp) start $name seeds 12 and 13; free $(avail) MB available"
  for k in 12 13; do
    bash -c "OMP_NUM_THREADS=2 PYTHONPATH=$ARCHIVE python3 $ARCHIVE/runs/reference8.py $LOGS/$name.log best 64 200 $radii $k > $LOGS/reference8-$name-best-seed$k.log 2> $LOGS/reference8-$name-best-seed$k.err < /dev/null" &
  done
  wait
done <<< "$(runs)"
echo "$(stamp) launch-ref8-var: the six runs ended"
