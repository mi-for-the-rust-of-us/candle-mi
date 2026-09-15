#!/usr/bin/env bash
# Newline activation patching (Figure-13 paper, arXiv revision, follow-up 1).
# Spec: BlackboxNLP 2026/Figure-13/docs/patching-spec.md (registered 2026-09-14).
#
# Two phases, in this order, because the second is only meaningful on prompts
# that pass the first:
#
#   PHASE=validate  baseline only, 60 lines per recipient prompt. Decides which
#                   prompts rhyme well enough to be usable at all.
#   PHASE=grid      the full experiment on the pairs named in $PAIRS, across
#                   three seeds: layer trace, all-layer patch, single-layer
#                   patch, identity control.
#
# Run from the repo root:
#   PHASE=validate bash scripts/run_patching.sh
#   PAIRS="p1-a-to-b p1-b-to-a" PHASE=grid bash scripts/run_patching.sh
# Print the commands without running them:
#   DRY_RUN=1 PHASE=grid bash scripts/run_patching.sh
#
# Resumable: a run whose output JSON already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

BIN="target/release/examples/figure13_newline_patch.exe"
[ -x "$BIN" ] || BIN="target/release/examples/figure13_newline_patch"
OUT="docs/experiments/figure13-patching"
LOG="$OUT/run.log"
PHASE="${PHASE:-validate}"
DRY_RUN="${DRY_RUN:-0}"
K="${K:-60}"
MODELS="${MODELS:-google/gemma-2-2b meta-llama/Llama-3.2-1B}"
PAIRS="${PAIRS:-p1-a-to-b p1-b-to-a p2-a-to-b p2-b-to-a p3-a-to-b p3-b-to-a p4-a-to-b p4-b-to-a}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p "$OUT"

[ -x "$BIN" ] || {
  echo "missing $BIN; build with:"
  echo "  cargo build --release --features transformer,mmap --example figure13_newline_patch"
  exit 1
}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# slug <model> -> a filename-safe short name
slug() { echo "$1" | sed 's|.*/||; s/[^A-Za-z0-9.-]/_/g'; }

# run <output> <args...>
run() {
  local out="$1"; shift
  if [ -f "$out" ]; then log "skip (exists) $out"; return; fi
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "$BIN" "$@" --output "$out"; echo
    return
  fi
  local t0=$(date +%s)
  "$BIN" "$@" --output "$out" 2>"${out%.json}.stderr"
  log "done  $out  ($(( $(date +%s) - t0 )) s)"
}

log "### patching $PHASE start (K=$K, DRY_RUN=$DRY_RUN) ###"

case "$PHASE" in
  validate)
    # One baseline per (model, pair-direction): the recipient's own rhyme rate.
    for m in $MODELS; do
      for p in $PAIRS; do
        run "$OUT/validate_$(slug "$m")_${p}.json" \
          --model "$m" --pair "$p" --k-samples "$K" --seed 1 --baseline-only
      done
    done
    log "### validation done ###  classify with:"
    log "  python <Figure-13>/scripts/patch_stats.py $OUT/validate_*.json"
    ;;
  grid)
    for m in $MODELS; do
      for p in $PAIRS; do
        for s in $SEEDS; do
          run "$OUT/patch_$(slug "$m")_${p}_s${s}.json" \
            --model "$m" --pair "$p" --k-samples "$K" --seed "$s"
        done
      done
    done
    log "### grid done ###  analyse with:"
    log "  python <Figure-13>/scripts/patch_stats.py $OUT/patch_*.json"
    ;;
  controls)
    # Registered controls (patching-spec.md, criterion 3).
    #
    #   same-rime : the donor's line 3 ends in the RECIPIENT's own rime, so the
    #               patch carries a different sentence but the same rhyme. It
    #               must not move the donor-group fraction; if it does, the
    #               instrument moves rhymes for reasons unrelated to the plan.
    #   mid-line  : the same donor row patched INSIDE line 3 instead of at the
    #               newline, which localizes any effect found.
    #
    # Same-rime donors keep lines 1-2 identical and swap line 3's final word for
    # a CMU rime-mate of the recipient's (verified 2026-09-15).
    G1=$'The stars were twinkling in the night,
The lanterns cast a golden light.
'
    G2=$'A sailor sailed across the bay,
And dreamed of home throughout the day.
'
    G4=$'The sun goes up, the sun goes down,
The moon shines bright above the town.
'

    same_rime() {  # pair  donor-line3  donor-word
      local pair="$1" line3="$2" word="$3" head
      case "$pair" in
        p1-*) head="$G1" ;;
        p2-*|p3-*) head="$G2" ;;
        p4-*) head="$G4" ;;
      esac
      run "$OUT/samerime_$(slug "$MODEL1")_${pair}.json"         --model "$MODEL1" --pair "$pair" --k-samples "$K" --seed 1         --donor-prompt "${head}${line3}"$'
' --donor-word "$word"
    }

    MODEL1="${MODEL1:-google/gemma-2-2b}"
    same_rime p1-a-to-b 'She wandered in the dark, unknown,'    unknown
    same_rime p1-b-to-a 'She wandered in the dark without,'     without
    same_rime p2-b-to-a 'The world keeps spinning even though,' though
    same_rime p3-a-to-b 'He raised his voice and gave a cry,'   cry
    same_rime p3-b-to-a 'He raised his voice and shouted out,'  out
    same_rime p4-a-to-b 'Nobody knows or remembers then,'       then
    same_rime p4-b-to-a 'Nobody knows or remembers you,'        you

    # mid-line: the real donor, patched 6 tokens before the newline.
    for p in $PAIRS; do
      run "$OUT/midline_$(slug "$MODEL1")_${p}.json"         --model "$MODEL1" --pair "$p" --k-samples "$K" --seed 1 --patch-offset 6
    done
    log "### controls done ###  analyse with:"
    log "  python <Figure-13>/scripts/patch_stats.py $OUT/samerime_*.json $OUT/midline_*.json"
    ;;
  *)
    echo "unknown PHASE '$PHASE' (expected: validate, grid, controls)"; exit 2
    ;;
esac
