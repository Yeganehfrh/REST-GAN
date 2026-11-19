#!/bin/sh
# Decode sex, age in years and age group in the 607 participants of ds005385 (Section 2.5; Table 2 and
# Supplementary Section S3), and fit the model to the same participants (Section 2.4.3). Run from the repository
# root; a CPU node is enough.
#
#   sh jobs/run_decoding.sh                      -> logs/decoding
#
# Settings (environment variables):
#   PY            python (default: python)
#   JOBS          processes for the cross-validation (default: the CPUs of the job, or 8)
#   MAIN          checkpoints of the three models
#   DATA, DEMOG   recordings (128 Hz, not preprocessed) and demographics of ds005385; the 607 participants are
#                 stored in two files each (285 and 322 participants)
#   CBRAMOD_DIR   directory with the code and the pretrained weights of CBraMod (pipeline/cbramod_features.py)
#   EDF_DIR       directory with the recordings of ds005385 (.edf), for the CBraMod features
#   OUT           output directory (default: logs/decoding)
#
# Steps, each with its log in the output directory:
#   0  CBraMod features of every segment, with three readouts (pipeline/cbramod_features.py)
#   1  features of every 4-s segment (pipeline/decode.py extract)
#   2  5-fold cross-validation over participants, 10 repeats (decode.py cv): the fixed feature sets, and the
#      readouts of the critic and of CBraMod chosen inside every training fold
#   3  scores, intervals and comparisons, with Holm's correction of the eight main comparisons (decode.py report)
#   4  embeddings fitted to the first 22 segments of every participant, generated data compared with the last 22
#      (pipeline/enrol_external.py)
set -e
PY=${PY:-python}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
OUT=${OUT:-logs/decoding}
MAIN=${MAIN:-"logs/main_s1_epoch_100.pt logs/main_s2_epoch_100.pt logs/main_s3_epoch_100.pt"}
DATA=${DATA:-"data/ds005385/ds005385.nc5 data/ds005385/ds005385_confirmation.nc5"}
DEMOG=${DEMOG:-"data/ds005385/demographic.csv data/ds005385/demographic_confirmation.csv"}
CBRAMOD_DIR=${CBRAMOD_DIR:-../CBraMod-benchmark}
EDF_DIR=${EDF_DIR:-data/ds005385/raw}
export PYTHONDONTWRITEBYTECODE=1
F=$OUT/features.npz
checksum() { sha256sum "$@" 2>/dev/null || shasum -a 256 "$@"; }
step() { echo "== $1 $(date)"; }
run() { log=$1; shift; "$@" > "$log" 2>&1 || { tail -5 "$log"; exit 1; }; }

for f in $DATA $DEMOG $MAIN; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
mkdir -p "$OUT"
{ echo "started: $(date)"; echo "host: $(hostname)"
  echo "commit: $(git rev-parse HEAD) ($(git describe --tags --always --dirty))"
  echo "files:"; checksum $DATA $DEMOG $MAIN; } > "$OUT/provenance.txt"

# one option per data file
DATA_ARGS=""; SUBJECT_ARGS=""; DEMOG_ARGS=""
for f in $DATA; do DATA_ARGS="$DATA_ARGS --data $f"; SUBJECT_ARGS="$SUBJECT_ARGS --subjects-from $f"; done
for f in $DEMOG; do DEMOG_ARGS="$DEMOG_ARGS --demographics $f"; done

# feature sets of the critics: critic_s<i> is the covariance of the second layer, critic_s<i>:mean the time averages
ITEMS=""; CRITICS=""; N_MAIN=0
for ckpt in $MAIN; do N_MAIN=$((N_MAIN + 1)); ITEMS="$ITEMS critic_s$N_MAIN=$ckpt"; CRITICS="$CRITICS critic_s$N_MAIN"; done
# the nine readouts of Supplementary Table S4, in its order
READOUTS=":mean :mean_all :meanstd_all :logvar_all :meanstd_h1 :meanstd_h2 :meanstd_h3 :cov_h1 _"
choice() {  # choice PREFIX NAME: the set PREFIXNAME:choice, chosen in every fold among the nine readouts of one critic
  alts=""; for r in $READOUTS; do [ "$r" = _ ] && r=""; alts="$alts|$1$2$r"; done; echo "$1$2:choice=${alts#|}"; }
sets_of() {  # sets_of PREFIX SUFFIX: one set per model, separated by commas
  s=""; for n in $CRITICS; do s="$s,$1$n$2"; done; echo "${s#,}"; }

# SECTION Run pipeline
step "0 CBraMod features"
if [ ! -f "$OUT/cbramod_pooled.npy" ]; then
  [ -f "$CBRAMOD_DIR/pretrained_weights/pretrained_weights.pth" ] \
    || { echo "missing: the code and the pretrained weights of CBraMod in $CBRAMOD_DIR (CBRAMOD_DIR)"; exit 1; }
  run "$OUT/0_cbramod.log" $PY -m src.EEGModalNet.pipeline.cbramod_features \
    --cbramod "$CBRAMOD_DIR" --edf-dir "$EDF_DIR" $SUBJECT_ARGS --out "$OUT/cbramod" --full
fi

step "1 features"
[ -f "$F" ] || run "$OUT/1_extract.log" $PY -m src.EEGModalNet.pipeline.decode extract $DATA_ARGS $DEMOG_ARGS \
  --extra-readouts --plain cbramod=$OUT/cbramod_pooled.npy --plain cbramod_ch=$OUT/cbramod_channels.npy \
  --plain cbramod_full=$OUT/cbramod_full.npy --plain-subjects $OUT/cbramod_subjects.npy --out "$F" $ITEMS

step "2 cross-validation ($JOBS processes)"
SETS="bandpower,filterbank,filterbank+bandpower,xspec,filterbank+xspec,cbramod,cbramod_ch"
SETS="$SETS,cbramod:choice=cbramod|cbramod_ch|cbramod_full"
for n in $CRITICS; do
  SETS="$SETS,$n,$n:mean,filterbank+$n,filterbank+$n:mean,$(choice '' $n),$(choice filterbank+ $n)"
done
run "$OUT/2_cv.log" $PY -m src.EEGModalNet.pipeline.decode cv --features "$F" --sets "$SETS" --jobs "$JOBS" \
  --out "$OUT/cv.jsonl"

step "3 report"
# --holm: the eight main comparisons (four for sex and four for age in years); --compare: further comparisons
run "$OUT/3_report.log" $PY -m src.EEGModalNet.pipeline.decode report --features "$F" --results "$OUT/cv.jsonl" \
  --critic $(sets_of '' :choice | tr ',' ' ') \
  --holm "$(sets_of '' :choice)" bandpower \
  --holm "$(sets_of '' :choice)" filterbank \
  --holm "$(sets_of '' :choice)" cbramod:choice \
  --holm "$(sets_of filterbank+ :choice)" filterbank \
  --compare "$(sets_of '' :mean)" bandpower \
  --compare "$(sets_of '' :mean)" cbramod_ch \
  --compare "$(sets_of filterbank+ :mean)" filterbank \
  --compare filterbank+bandpower filterbank \
  --compare filterbank+xspec filterbank \
  --compare "$(sets_of '' :choice)" "$(sets_of '' :mean)" \
  --compare cbramod_ch cbramod \
  --out "$OUT/report"

step "4 fitting the model to the participants"
E=$OUT/enrolment; mkdir -p "$E"
if [ ! -f "$E/metrics.json" ]; then
  T=$((JOBS / N_MAIN)); [ "$T" -ge 1 ] || T=1          # the models are fitted side by side
  for ckpt in $MAIN; do
    OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T $PY -m src.EEGModalNet.pipeline.enrol_external "$ckpt" $DATA_ARGS --out "$E" \
      --fit-only > "$E/fit_$(basename "$ckpt" .pt).log" 2>&1 &
  done
  wait
  run "$OUT/4_enrolment.log" $PY -m src.EEGModalNet.pipeline.enrol_external $MAIN $DATA_ARGS --out "$E"
fi

echo "finished: $(date)" >> "$OUT/provenance.txt"
step "DONE: $OUT/report/report.md, $E/report.md"
