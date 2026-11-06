#!/usr/bin/env bash
# Download XAUUSD M15 bid + ask candles from Dukascopy in yearly chunks, then merge.
# Requires Node.js (npx). Re-running skips chunks that already exist.
# Usage: scripts/fetch_data.sh [first_year]   (default 2021)
set -euo pipefail
cd "$(dirname "$0")/.."

FIRST_YEAR="${1:-2021}"
LAST_YEAR="$(date -u +%Y)"
RAW=data/raw
mkdir -p "$RAW"

for y in $(seq "$FIRST_YEAR" "$LAST_YEAR"); do
  to="$((y + 1))-01-01"
  [ "$y" = "$LAST_YEAR" ] && to=now
  for side in bid ask; do
    f="xauusd_m15_${side}_${y}"
    if [ -s "$RAW/$f.csv" ] && [ "$y" != "$LAST_YEAR" ]; then
      echo "skip $f"; continue
    fi
    # Note: do not use --retry-on-empty; weekends/holidays are legitimately empty.
    npx -y dukascopy-node -i xauusd -from "${y}-01-01" -to "$to" -t m15 -p "$side" \
      -f csv -v -dir "$RAW" -fn "$f" -r 5 -rp 2000 -s
  done
done

python -m goldbot.data.dukascopy "$RAW" xauusd_m15 "data/xauusd_m15_${FIRST_YEAR}_${LAST_YEAR}.csv"
