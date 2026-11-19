#!/bin/sh
# Decode sex, age in years and age group from the critics of the ablation variants (Figure 4 and Supplementary
# Section S4), in the 607 participants of ds005385. Run after jobs/run_decoding.sh: its features and results are
# continued in copies. Run from the repository root; a CPU node is enough.
#
#   sh jobs/run_decoding_ablations.sh            -> logs/decoding_ablations
#
# Settings (environment variables): PY, JOBS, DATA and DEMOG as in jobs/run_decoding.sh, and
#   VARIANTS      LABEL=checkpoint for every variant of Table 1
#   FILM          checkpoint of the variant with FiLM at the critic's input (Supplementary Section S4). It is read
#                 out with three stand-ins for the participant: mean embedding, zero vector, FiLM step removed
#   BASE          output directory of jobs/run_decoding.sh (default: logs/decoding)
#   OUT           output directory (default: logs/decoding_ablations)
set -e
PY=${PY:-python}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
BASE=${BASE:-logs/decoding}
OUT=${OUT:-logs/decoding_ablations}
DATA=${DATA:-"data/ds005385/ds005385.nc5 data/ds005385/ds005385_confirmation.nc5"}
DEMOG=${DEMOG:-"data/ds005385/demographic.csv data/ds005385/demographic_confirmation.csv"}
VARIANTS=${VARIANTS:-"A1=logs/A1_s1_epoch_100.pt A2=logs/A2_s1_epoch_100.pt A3=logs/A3_s1_epoch_100.pt
  A4=logs/A4_s1_epoch_100.pt A5=logs/A5_s1_epoch_100.pt A6=logs/A6_s1_epoch_100.pt A7=logs/A7_s1_epoch_100.pt
  A8=logs/A8_s1_epoch_100.pt G1=logs/G1_s1_epoch_100.pt G2=logs/G2_s1_epoch_100.pt"}
FILM=${FILM:-logs/film_s1_epoch_100.pt}
export PYTHONDONTWRITEBYTECODE=1
F=$OUT/features.npz
step() { echo "== $1 $(date)"; }
run() { log=$1; shift; "$@" > "$log" 2>&1 || { tail -5 "$log"; exit 1; }; }

for f in $DATA $DEMOG $FILM "$BASE/features.npz" "$BASE/cv.jsonl"; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
for item in $VARIANTS; do [ -f "${item#*=}" ] || { echo "missing: ${item#*=}"; exit 1; }; done
mkdir -p "$OUT"

# one option per data file
DATA_ARGS=""; DEMOG_ARGS=""
for f in $DATA; do DATA_ARGS="$DATA_ARGS --data $f"; done
for f in $DEMOG; do DEMOG_ARGS="$DEMOG_ARGS --demographics $f"; done

# SECTION Run pipeline
step "1 features of the variants"
ITEMS="$VARIANTS film_mean=$FILM@mean film_zero=$FILM@zero film_none=$FILM@none"
if [ ! -f "$F.done" ]; then
  cp "$BASE/features.npz" "$F"
  run "$OUT/1_extract.log" $PY -m src.EEGModalNet.pipeline.decode extract $DATA_ARGS $DEMOG_ARGS --out "$F" $ITEMS
  touch "$F.done"
fi

step "2 cross-validation ($JOBS processes)"
# both readouts of every variant: covariance of the second layer (NAME) and time averages (NAME:mean)
SETS=""; for item in $ITEMS; do n=${item%%=*}; SETS="$SETS,$n,$n:mean"; done; SETS=${SETS#,}
[ -f "$OUT/cv.jsonl" ] || cp "$BASE/cv.jsonl" "$OUT/cv.jsonl"
run "$OUT/2_cv.log" $PY -m src.EEGModalNet.pipeline.decode cv --features "$F" --sets "$SETS" --jobs "$JOBS" \
  --out "$OUT/cv.jsonl"

step "3 report"
# --holm: the stand-ins for the participant against the mean embedding, for the three targets
run "$OUT/3_report.log" $PY -m src.EEGModalNet.pipeline.decode report --features "$F" --results "$OUT/cv.jsonl" \
  --critic critic_s1 critic_s2 critic_s3 \
  --holm film_zero film_mean --holm film_none film_mean --holm-targets sex age age_group \
  --out "$OUT/report"
step "DONE: $OUT/report/report.md"
