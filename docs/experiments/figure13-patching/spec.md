# Newline activation patching: specification and registered criteria

**Status**: specification, written 2026-09-14 before any run. Everything above
the "Validation record" heading was fixed before the harness existed and is not
edited afterwards; that section and anything below it is written after the fact
and says so.
**Paper**: arXiv revision of "Planning or Improvisation?". Answers §7 reading
(ii), "CLT-invisible planning", and reviewer yeTd's suggestion 5.
**Harness**: a new `candle-mi/examples/figure13_newline_patch.rs`.
**Analysis**: `Figure-13/scripts/patch_stats.py` (to be written), reusing the
phonology and insertion checks of `scripts/horizon_leakage.py`.

## Why this experiment and not another

Every causal probe in the paper so far writes a **cross-layer-transcoder
decoder direction** into the residual stream. Two objections survive that
design, and neither can be answered from inside it:

1. **Reading (ii), CLT-invisible planning.** Our features are discovered
   bottom-up from decoder vectors, so they are "say $X$" directions. A plan
   held in a form those directions cannot engage is invisible to every
   experiment we have run.
2. **The instrument, not the model.** A correspondent (FAR.AI, by email) put it
   that the suppress-and-inject protocol is activation patching by another
   name. It is an activation intervention, but a *feature-basis, additive* one:
   it adds a direction we had to find first.

Activation patching *replaces* a residual row with one the model itself
produced on another prompt. It needs no transcoder, no feature discovery, and
no choice of strength. It is therefore the one instrument in reach whose null
result cannot be explained by our feature-discovery method, and its positive
result would show that the newline carries a rhyme plan our features missed.

A second consequence, deferred to v2 and recorded here so the decision is on
file: because no CLT is involved, the same harness runs on any model that fits
in 16 GB, including models with no open CLT. That is the only route we have to
the scale reading (§7 (i)). v1 is deliberately scoped to the two behavioural
cells.

## Design

### Minimal-pair prompts

Patching the last prompt position transfers that position's entire state, not
just a rhyme. With arbitrary donor and recipient prompts a rhyme change would
therefore be uninformative: the recipient would simply have inherited the
donor's context. We use **minimal pairs**: two prompts identical through line 3
except for line 3's final word, which sets a different rime. Everything the
patch could transfer is then shared between donor and recipient except the
rhyme, so a rhyme change is attributable to it.

All four prompts are AABB: lines 1 and 2 rhyme with each other, line 3 opens a
new couplet, and line 4 is expected to rhyme with line 3. The monorhyme prompts
used elsewhere in the paper are unsuitable here, because changing line 3's
ending would leave lines 1 and 2 pulling toward the original rime.

| Pair | Shared lines 1--2 | Member A (rime) | Member B (rime) |
|---|---|---|---|
| P1 | The stars were twinkling in the night, / The lanterns cast a golden light. | She wandered in the dark about, (AW1 T) | She wandered in the dark alone, (OW1 N) |
| P2 | A sailor sailed across the bay, / And dreamed of home throughout the day. | The world keeps spinning even so, (OW1) | The world keeps spinning even now, (AW1) |
| P3 | A sailor sailed across the bay, / And dreamed of home throughout the day. | He raised his voice and gave a shout, (AW1 T) | He raised his voice and gave a sigh, (AY1) |
| P4 | The sun goes up, the sun goes down, / The moon shines bright above the town. | Nobody knows or remembers who, (UW1) | Nobody knows or remembers when, (EH1 N) |

Each pair is run in **both directions** (A donates to B, B donates to A), which
doubles the data and provides a symmetry check: a mechanism that transfers a
rhyme plan should work in both directions.

**Prompt validation, before the experiment.** Each of the eight prompts must
elicit a line-3-rhyming line 4 at a rate distinguishable from zero on each
model, measured as the natural-group rate over 60 unpatched samples. A prompt
failing this on a model is dropped for that model and the drop is recorded;
patching cannot show a rhyme moving in a prompt whose rhyme does not hold.

### Conditions

Recipient B, donor A, patch applied at B's line-3 newline, which is the final
prompt token. The patch is re-applied at that position at every generation step
(candle-mi is KV-cache-free; `route: "recompute-per-step"`, matching the
composition-horizon harness).

| Condition | Patch |
|---|---|
| `baseline` | none: B unpatched |
| `single-layer` | donor A's `ResidPost(L)` row at the newline, one layer L, swept over all layers |
| `all-layer` | donor A's newline row at **every** layer simultaneously |
| `same-rime` control | donor is a third prompt whose line 3 ends in B's own rime; expected to do nothing |
| `identity` control | donor is B itself; must be an exact no-op |
| `mid-line` control | A's residual patched at a mid-line position of B instead of the newline |

The `all-layer` condition is registered because a null under single-layer
patching admits the escape "the plan is distributed across layers". Patching
every layer at that position replaces the newline's entire state, which is the
strongest transfer the geometry allows; a null there closes the escape.

### Readouts

Per condition, matching the composition-horizon experiment so the two are
directly comparable:

- **m1** rime class of the composed line's final word over 60 sampled lines
  (temperature 0.7, seeds 1, 2, 3), classified as **donor group** (A's rime),
  **recipient group** (B's own rime), or other.
- **m2** the greedy line, verbatim.
- **m3** teacher-forced probability of the donor's and the recipient's rhyme
  words at the final-word slot of B's unpatched greedy line.
- **m4** the layer sweep of m3, the causal trace.
- **insertion** whether the donor's line-3 final word, or its first content
  word, appears in the composed line and where. The composition-horizon runs
  showed that an intervention at this position captures the next token almost
  deterministically, so any rhyme effect must be reported alongside this.

## Registered criteria

Let N be the pooled sampled lines per condition (60 lines x 3 seeds x the
surviving prompts x 2 directions).

1. **H1, the newline carries a rhyme plan.** Under `single-layer` at any layer,
   or under `all-layer`, the pooled donor-group fraction has a Clopper-Pearson
   95% lower bound above the unpatched baseline's upper bound, in both
   directions of at least one pair, and the effect is not explained by the
   insertion check. Reported per layer, per pair and per direction.
2. **H2, next-token capture without rhyme transfer.** The patch moves the first
   token of the composed line (donor's opening or rhyme word appearing at
   position 0) while H1 fails. This is the composition-horizon result
   reproduced without a transcoder.
3. **Controls.** `identity` must produce byte-identical greedy output and an m1
   within sampling noise; a failure invalidates the run. `same-rime` must not
   move the donor-group fraction. `mid-line` localizes any effect found.
4. **Minimum detectable effect** stated from N, as in §4.4.

### Predictions, written before running

H1 fails and H2 holds, in both models and all surviving pairs: the patch
captures the next token and the composed line still rhymes with the recipient's
own line 3. This follows from the composition-horizon result, but it is a
genuinely different instrument and the prediction is therefore falsifiable in a
way the earlier one was not.

If H1 holds anywhere, reading (ii) becomes the primary reading for that model,
§7 is rewritten around it, and the abstract changes: it would mean the newline
does carry a rhyme plan and our decoder-derived features simply could not
engage it. That outcome would strengthen rather than weaken the paper's
reproducibility message, since the failure would be located in the feature
discovery method the paper already names as load-bearing.

## Validation before any science

The CUDA path of `Intervention::PatchAt` was silently wrong until commit
`5f96e40` (2026-09-02), which reimplemented it as a masked select. That history
means the harness is validated before it is trusted:

1. **Identity patch** is a bit-exact no-op on both devices.
2. **Positive control**: in the next-token geometry (prompt truncated so the
   rhyme word is the next token), patching the final position from a donor with
   a different rhyme must move the next token toward the donor's word. If this
   fails, the harness is broken, not the hypothesis.
3. **CPU against GPU** on a small case, since the fixed bug was device-specific.
4. `HookPoint::accepts_positional_patch()` is consulted rather than assumed:
   `PatchAt` is accepted only where the activation is `[batch, seq, hidden]`.

## Cost

Stage 1, the layer sweep on m3, is one forward per layer per pair-direction:
seconds to minutes, no sampling. Stage 2 samples 60 lines only at the layers
stage 1 flags, plus `all-layer` and the controls. At the measured 1.55 s per
sampled line on Gemma and 0.34 s on Llama, stage 2 is roughly 2 hours on Gemma
and half an hour on Llama for the full grid of surviving prompts, directions
and seeds. Resumable, as before.

## Outputs

`candle-mi/docs/experiments/figure13-patching/`: `validate_*.json` (the four
checks above), `trace_<model>_<pair>_<dir>.json` (stage 1), and
`patch_<model>_<pair>_<dir>_s<seed>.json` (stage 2). Copied afterwards to
`Figure-13/data/patching/` with the canonical analysis printouts.

---

## Validation record (2026-09-14)

Everything above this line was written before the harness existed. This section
and anything below it was written afterwards.

**Harness.** `candle-mi/examples/figure13_newline_patch.rs`, registered in
`Cargo.toml` as a `transformer`-only example (no CLT feature: the instrument
uses no transcoder). Clean under `cargo fmt`, under
`cargo clippy --all-targets --no-default-features --features mmap,transformer
-- -W clippy::pedantic -D warnings`, and under the rustdoc lane. The crate's
own `registration_guard` test enforces the manifest entry.

**1. Identity control passes.** Patching the recipient with its own newline row
at all 26 layers of Gemma 2 2B leaves the greedy line byte-identical and moves
the tracked probabilities by at most 1.4e-8, against a 1e-6 threshold. The CUDA
path is therefore sound on this machine, which the pre-v0.1.24 history made
worth checking rather than assuming.

**2. Positive control passes, and is informative.** Patching from a *different
poem* (P4's newline into P1's recipient, not a minimal pair) changes the
composed line from "Her heart was heavy, her soul was gone." to "The moon was
shining on her face.", which is the donor's imagery. The instrument therefore
transfers a great deal. Under that same transfer the donor's rhyme word moves
from 3.4e-6 to 4.1e-6 and the line does not rhyme with it.

**3. Index alignment.** Both members of pair P1 tokenize to 26 tokens with the
newline at index 25 on Gemma, so donor and recipient rows are taken from the
same position. The harness does not assume this: it records each prompt's own
newline index and asserts the token contains a newline.

**Instrument change made after seeing smoke data, recorded as such.** The first
smoke run showed the minimal-pair patch barely moving the output, which is
ambiguous between "the newline does not encode the rhyme" and "the two rows are
so alike that the patch carried nothing". A per-layer **row-divergence
diagnostic** (cosine and relative L2 between the donor's and the recipient's
newline rows) was added to resolve that ambiguity. It is a diagnostic, not a
hypothesis test, and it does not touch the registered criteria. On pair P1 the
rows differ down to cosine 0.925, with relative L2 up to 0.39, so the newline
does encode the difference between a line ending "about" and one ending
"alone". Any null in H1 is therefore a null about what the model *does* with
that difference, not about whether it exists.

**Not yet run.** Prompt validation (the per-prompt unpatched rhyme rates that
decide which prompts survive), and the full grid of pairs, directions, seeds
and models.

---

## Scope and criterion amendments (2026-09-15, before the grid)

Written after the validation phase and before any grid run. The registered
criteria above are unedited; these are disclosed amendments to them.

**1. The usability threshold was too weak, and is replaced.** The registered
rule was that a prompt's unpatched rhyme rate be "distinguishable from zero",
which passes at 2/60. That is unusable here, and the right threshold follows
from the design rather than from taste: **the recipient's own rhyme rate is the
ceiling on any redirect**, because a patch can only move the rhyme of a line
that rhymes at all. With a near-zero donor baseline the minimum detectable
donor-group fraction is 0.150, so a prompt is usable only if its rhyme rate
exceeds that. A prompt below it could not show a redirect even if the transfer
were perfect.

**2. Llama 3.2 1B is dropped; v1 is Gemma 2 2B alone.** Measured unpatched
rhyme rates over 60 lines per prompt: Gemma 0.20 to 0.42 (one prompt at 0.05),
Llama 0.00 to 0.08. Every Llama prompt fails the amended threshold. Three
independent lines agree that this is structural rather than a bad prompt set:

- Llama's own validated prompts in `plip-rs/corpus/llama_prompts.json` are all
  monorhyme (tree/free/sea, shore/more/for, blue/dew/new, hat/sat/that). The
  minimal-pair design needs AABB, because in a monorhyme prompt changing line
  3's ending leaves lines 1 and 2 pulling toward the original rhyme.
- `plip-rs/docs/planning-circuit-hunt/03-cross-model.md`: "the behavioral
  experiments showed Gemma reliably produces rhyming couplets while Llama does
  not", with a mechanism (in Gemma a rhyme candidate persists through the final
  layers; in Llama the signal does not survive).
- The 60% figure in `plip-rs/outputs/llama_couplet_results.json` does **not**
  transfer: the repository's own index labels it an Ollama baseline, and its
  prompts are instruction-style, not base-model completions.

The paper must therefore say that patching answers the CLT-invisible-plan
question on the one open model that rhymes reliably enough to be asked, and
must not present a Gemma-only result as cross-model.

**3. Prompt p2-a-to-b is dropped** (rhyme rate 0.05, below the 0.150
threshold). Seven of eight pair-directions survive.

**4. Both missing controls are now implemented**, so criterion 3 can be
evaluated as registered. `same-rime` uses a third donor whose line 3 ends in a
CMU rime-mate of the recipient's own ending (unknown/without/though/cry/out/
then/you, each verified against CMUdict); it needs no new code, only the
existing `--donor-prompt` override. `mid-line` required a new `--patch-offset`
flag, which patches N tokens before the newline; at offset 6 it lands inside
line 3 (" wandered" on pair P1) and is recorded in the output as
`patch_offset` and `patch_position`.

**Run plan.** Grid: 7 pairs x 3 seeds, one already banked, 20 runs at 306 s =
1.7 h. Controls: 7 same-rime + 7 mid-line at seed 1 = 14 runs = 1.2 h. Total
about 2.9 h, resumable.

---

## Results (2026-09-15)

35 grid and control runs on Gemma 2 2B, plus 7 corrected mid-line reruns. No
spill in any run (`hmn watch --follow-new`). The **identity control passed in
all 42 runs**, maximum probability delta 1.4e-8.

**H1 fails: the newline carries no rhyme plan this instrument can find.**
Pooled over 7 prompts x 3 seeds, N = 1260 sampled lines per condition:

| Condition | donor-group endings | Fisher vs baseline |
|---|---|---|
| baseline | 4/1260 = 0.003 | |
| all-layer | 11/1260 = 0.009 | p = 0.12 |
| single-layer | 10/1260 = 0.008 | p = 0.18 |

Minimum detectable donor fraction 0.0143. Replacing the newline's entire state
at every layer with that of a poem whose line 3 ends on a different rhyme does
not move the rhyme the model then writes, and the design would have seen a
1.4% redirect. The recipient's own rhyme is also untouched: 384/1260 = 0.305
baseline against 357/1260 = 0.283 under the all-layer patch (p = 0.26).

**H2 holds, weakly.** The composed line opens with the donor's line-3 word in
33/1260 samples against 4/1260 at baseline (p = 9.2e-7); single-layer 18/1260
(p = 4.2e-3). Real next-token capture, but 2.6% where feature injection in the
composition-horizon experiment reached 97.6%. A minimal-pair patch simply
carries little, which is the point of using one.

**All three controls pass.** `identity` as above. `same-rime` (donor's line 3
ends in a CMU rime-mate of the recipient's) does not move the rhyme:
136/420 = 0.324 against 114/420 = 0.271, p = 0.11. `mid-line`, after the fix
below, is an exact no-op: 136/420 against 136/420, p = 1.000, donor-group
2/420 unchanged.

**Defect found and corrected before the mid-line control was believed.** The
first implementation captured the donor row at its *newline* and wrote it into
the recipient's *mid-line* position, and its progress line claimed otherwise.
It appeared to show that mid-line patching destroys the rhyme (0.324 to 0.110,
p = 1.7e-12) while newline patching does not, which would have been an
attractive localization result and was an artefact of writing an out-of-place
state. Under a minimal pair the mid-line rows are *identical* between donor and
recipient, so the control must be a no-op; that it was not is what exposed the
bug. Corrected to capture at the same offset it patches, it is now a no-op, and
that no-op is itself evidence the harness writes what it claims. The superseded
runs are kept in `candle-mi/docs/experiments/figure13-patching/superseded/`.
The grid is unaffected: at offset 0 the donor position *is* the newline.

**Also fixed:** `horizon_stats.binom_cdf` overflowed at n = 1260, raising
`OverflowError` rather than returning a wrong number. Rewritten in log space
and validated against SciPy to 14 decimal places; the paper's 17 verification
checks still pass.

**Reading.** Reading (ii) of §7, that a plan exists but our decoder-derived
features cannot engage it, loses its main support: an instrument that needs no
features, no transcoder and no strength finds nothing at the newline either.
The result is Gemma-only and must be reported as such.
