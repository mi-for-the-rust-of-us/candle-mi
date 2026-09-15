#!/usr/bin/env bash
# Composition-horizon power run (Figure-13 paper, arXiv revision, follow-up 2).
# Spec: BlackboxNLP 2026/Figure-13/docs/horizon-power-spec.md (registered
# 2026-09-11). Reruns figure13_newline_steering over three or four validated
# prompts per cell, 60 sampled lines per condition, three seeds, and adds the
# m1 classification in place after each run.
#
# Run from the repo root:   bash scripts/run_horizon_power.sh
# Print commands only:      DRY_RUN=1 bash scripts/run_horizon_power.sh
# Resumable: a run whose output JSON already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

BIN="target/release/examples/figure13_newline_steering.exe"
OUT="docs/experiments/figure13-newline/power"
LOG="$OUT/run.log"
K=60
SEEDS="1 2 3"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "missing $BIN; build with: cargo build --release --features clt,transformer,mmap --example figure13_newline_steering"; exit 1; }

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# run <output> <args...>: one harness invocation plus the m1 classifier.
run() {
  local out="$1"; shift
  if [ -f "$out" ]; then log "skip (exists) $out"; return; fi
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "$BIN" "$@" --output "$out"; echo
    echo "python scripts/newline_steering_classify.py $out"
    return
  fi
  local t0=$(date +%s)
  "$BIN" "$@" --output "$out" 2>"${out%.json}.stderr"
  python scripts/newline_steering_classify.py "$out" >/dev/null
  log "done  $out  ($(( $(date +%s) - t0 )) s)"
}

# Prompts: first three lines plus the line-3 newline, passed verbatim.
G_ABOUT=$'The stars were twinkling in the night,\nThe lanterns cast a golden light.\nShe wandered in the dark about,\n'
G_SO=$'A sailor sailed across the bay,\nAnd dreamed of home throughout the day.\nThe world keeps spinning even so,\n'
G_SHOUT=$'A sailor sailed across the bay,\nAnd dreamed of home throughout the day.\nHe raised his voice and gave a shout,\n'
G_WHO=$'The sun goes up, the sun goes down,\nThe moon shines bright above the town.\nNobody knows or remembers who,\n'
L_FREE=$'The birds were singing in the tree,\nAnd everything was wild and free.\nThe river ran down to the sea,\n'
L_NEW=$'The morning sky was painted blue,\nThe garden sparkled bright with dew.\nThe world had started fresh and new,\n'
L_MORE=$'The waves came crashing on the shore,\nThe wind was howling more and more.\nShe asked what all the fuss was for,\n'

# Suppress feature sets: every feature of the natural rhyme group (spec table).
sf() { local a=(); for f in "$@"; do a+=(--suppress-feature "$f"); done; echo "${a[@]}"; }
G426_OUT=$(sf 16:13725 25:9385)
G426_OW=$(sf 25:6778 25:4985 25:4505 22:10362 25:5776 20:12770 19:3248)
G426_OO=$(sf 25:10073 23:1548 25:14014 18:7484 19:5076 23:3304 25:5927)
G25_OUT=$(sf 25:57092 23:49923 20:77102)
G25_OW=$(sf 25:70598 25:51326 25:52789 25:28986 25:5076 25:73247 25:97279 25:94279 24:78290 21:89350 21:51174 18:7726 23:94103)
G25_OO=$(sf 22:86352 25:46148 25:70439 23:15981 19:39669 22:72623 18:46523 25:75246 19:70754)
L_EE=$(sf 13:30985 9:5488 14:27874 13:32049)
L_OO=$(sf 14:18284 15:8165 11:20779)
L_ORE=$(sf 1:5297 3:22663 10:18203)

log "### horizon power run start (K=$K, seeds: $SEEDS, DRY_RUN=$DRY_RUN) ###"

# ── Step 0: stack replication of the July runs (prompt 1, k=20, seed 42) ─────
run "$OUT/replicate_gemma2-2b-426k.json"      --preset gemma2-2b-426k      --strength 25 --k-samples 20 --seed 42
run "$OUT/replicate_llama3.2-1b-524k.json"    --preset llama3.2-1b-524k    --strength 25 --k-samples 20 --seed 42
run "$OUT/replicate_gemma2-2b-2.5m.json"      --preset gemma2-2b-2.5m      --strength 10 --k-samples 20 --seed 42
run "$OUT/replicate_qwen3-0.6b-16k-ation.json" --preset qwen3-0.6b-16k-ation --strength 25 --k-samples 20 --seed 42
log "### step 0 done; compare with: python <Figure-13>/scripts/horizon_stats.py --compare <july.json> <replicate.json> ###"

# ── Power runs, shortest cell first ──────────────────────────────────────────
for s in $SEEDS; do
  P="llama3.2-1b-524k"
  run "$OUT/fullline_${P}_free_s$s.json" --preset $P --strength 25 --k-samples $K --seed $s --prompt "$L_FREE" --suppress-word free $L_EE
  run "$OUT/fullline_${P}_new_s$s.json"  --preset $P --strength 25 --k-samples $K --seed $s --prompt "$L_NEW"  --suppress-word new  $L_OO
  run "$OUT/fullline_${P}_more_s$s.json" --preset $P --strength 25 --k-samples $K --seed $s --prompt "$L_MORE" --suppress-word more $L_ORE
done

for s in $SEEDS; do
  P="gemma2-2b-426k"
  run "$OUT/fullline_${P}_about_s$s.json" --preset $P --strength 25 --k-samples $K --seed $s --prompt "$G_ABOUT" --suppress-word about $G426_OUT
  run "$OUT/fullline_${P}_so_s$s.json"    --preset $P --strength 25 --k-samples $K --seed $s --prompt "$G_SO"    --suppress-word so    $G426_OW
  run "$OUT/fullline_${P}_shout_s$s.json" --preset $P --strength 25 --k-samples $K --seed $s --prompt "$G_SHOUT" --suppress-word shout $G426_OUT
  run "$OUT/fullline_${P}_who_s$s.json"   --preset $P --strength 25 --k-samples $K --seed $s --prompt "$G_WHO"   --suppress-word who   $G426_OO
done

for s in $SEEDS; do
  P="gemma2-2b-2.5m"
  run "$OUT/fullline_${P}_about_s$s.json" --preset $P --strength 10 --k-samples $K --seed $s --prompt "$G_ABOUT" --suppress-word about $G25_OUT
  run "$OUT/fullline_${P}_so_s$s.json"    --preset $P --strength 10 --k-samples $K --seed $s --prompt "$G_SO"    --suppress-word so    $G25_OW
  run "$OUT/fullline_${P}_shout_s$s.json" --preset $P --strength 10 --k-samples $K --seed $s --prompt "$G_SHOUT" --suppress-word shout $G25_OUT
  run "$OUT/fullline_${P}_who_s$s.json"   --preset $P --strength 10 --k-samples $K --seed $s --prompt "$G_WHO"   --suppress-word who   $G25_OO
done

for s in $SEEDS; do
  P="qwen3-0.6b-16k-ation"
  run "$OUT/fullline_${P}_ation_s$s.json" --preset $P --strength 25 --k-samples $K --seed $s
done

log "### ALL DONE ###  analyse with: python <Figure-13>/scripts/horizon_stats.py --pool $OUT/fullline_*.json"
