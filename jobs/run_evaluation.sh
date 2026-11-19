#!/bin/sh
# Evaluate the generated EEG of trained models (Section 2.4). Run from the repository root; a CPU is enough.
#
#   sh jobs/run_evaluation.sh logs/main_s1_epoch_100.pt [more checkpoints]
#   ENROL=0 sh jobs/run_evaluation.sh <checkpoints>         without fitting embeddings to the held-out participants
#   LIVE=1 sh jobs/run_evaluation.sh <checkpoints>          the generator's weights as trained, not their running average
#   MAIN="<dir> <dir> ..." sh jobs/run_evaluation.sh <ablation checkpoints>
#
# Every checkpoint is evaluated with pipeline/evaluate.py (three generation seeds) and written to
# $OUT/<run>_epoch_<N>/. Then the tables are built with pipeline/aggregate_eval.py: the checkpoints are taken as
# training seeds of one model, unless MAIN lists the evaluation directories of the full model, in which case
# the checkpoints are its ablations.
set -e
PY=${PY:-python}
OUT=${OUT:-logs/eval}
ENROL_FLAG=""; [ "${ENROL:-1}" = 1 ] || ENROL_FLAG="--skip-enrolment"
LIVE_FLAG=""; SUFFIX=""; if [ "${LIVE:-0}" = 1 ]; then LIVE_FLAG="--live-weights"; SUFFIX="_live"; fi
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
[ $# -ge 1 ] || { echo "usage: sh jobs/run_evaluation.sh <checkpoint.pt> [...]"; exit 1; }

# SECTION Evaluate the checkpoints
DIRS=""
for ckpt in "$@"; do
  name=$(basename "$ckpt" .pt)$SUFFIX
  if [ -f "$OUT/$name/metrics.json" ]; then echo "== $name: already evaluated"; else
    echo "== $name $(date)"
    $PY -m src.EEGModalNet.pipeline.evaluate "$ckpt" --gen-seeds 3 --out "$OUT" $ENROL_FLAG $LIVE_FLAG \
      > "$OUT/$name.log" 2>&1 || { tail -5 "$OUT/$name.log"; exit 1; }
    grep -E "four feature spaces" "$OUT/$name.log" || true
  fi
  DIRS="$DIRS $OUT/$name"
done

# SECTION Tables
echo "== tables $(date)"
if [ -n "${MAIN:-}" ]; then
  ABLATIONS=""; for d in $DIRS; do ABLATIONS="$ABLATIONS $(basename "$d")=$d"; done
  $PY -m src.EEGModalNet.pipeline.aggregate_eval --main $MAIN --ablation $ABLATIONS --out "$OUT/tables"
else
  $PY -m src.EEGModalNet.pipeline.aggregate_eval --main $DIRS --out "$OUT/tables"
fi
echo "== DONE $(date)"
