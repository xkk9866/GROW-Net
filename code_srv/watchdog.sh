#!/bin/bash
# kill teacher_ab if its log has not changed for 40 minutes (engine-core crash leaves the parent hanging); q9t.sh retries/resumes
cd ~/research/grow_www27
while ! grep -q Q9T_DONE logs/q9t.log 2>/dev/null; do
  sleep 300
  for f in logs/teacherab_*.log; do
    [ -f "$f" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ $age -gt 2400 ] && pgrep -f teacher_ab.py > /dev/null; then echo "$(date) stall $f -> kill"; pkill -9 -f teacher_ab.py; fi
  done
done
