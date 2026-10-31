# `clt_train_tiny` — a complete CLT training recipe, and the CLT loose ends — Plan for v0.2.3

**Scope:** one example (`examples/clt_train_tiny.rs`) plus the loose ends the merged `clt-train` work left
behind, all for **v0.2.3**. No new library API: everything the example calls already exists
(`clt::train`, `clt::replacement`, `optim::AdamW`, `CrossLayerTranscoder::open_local`).

**Written** 2026-10-31 in an hf-fetch-model session, for a fresh candle-mi session. Nothing in this plan has been
implemented; every fact below was read from this repository on that date (file and line references are to
`main` at `7b1fac8`, four local commits ahead of `origin/main`). Re-read a line before relying on it.

**Validated** 2026-10-31 in a second pass: every file and line reference was re-read, and the recipe was compiled and
run as a throwaway spike (CPU, 5.2 s, since deleted). The corrections it produced are folded in below; the numbers
are recorded under "Validation record".

**Status 2026-10-31: implemented on `main`, not pushed.** The example (`examples/clt_train_tiny.rs`), its CI lane
(`clt-train`: clippy, rustdoc, build, run, in `ci.yml`, `publish.yml` and `scripts/preflight.ps1`), S1, S2 and S3 are
done, in five commits. D4 to D8 went as recommended; D6 (no root re-exports) is recorded in
`docs/roadmaps/v0.3.0-api-ergonomics.md`. Still open and not candle-mi work: S4 (the two upstream `candle` issues, a
maintainer decision after re-measuring) and the parked `candle` 0.11.1 move. Still to do before v0.2.3: the release
ritual in `CLAUDE.md` (`preflight.ps1 -Ci` on the full tree, version bump, dry-run publish). Measured on the first
run: the new lint lane found nothing to fix in the two existing test files, contrary to the expectation below.

**Source of the ask:** [`docs/dogfooding-feedbacks/trainable-clt-and-the-candle-costs-of-training-it.md`](../dogfooding-feedbacks/trainable-clt-and-the-candle-costs-of-training-it.md)
§8 item 3, "An example, `clt_train_tiny`: a complete loop on a small model, as a recipe — the loop stays out of the
library, but a reader should not have to find canvas to learn how the pieces fit." It is the only item of that
report's v0.2.3 ask that is still open: items 1 and 2 shipped as `clt::replacement` (a variant of what §8 sketched),
and §2 to §5 shipped as `clt::train`.

---

## What is at stake

1. **A recipe that rots.** An example that needs a model download and a GPU is never run by CI, and drifts from the
   API it teaches. This one must run **offline, on CPU, in seconds**, and CI must **run** it, not only build it.
2. **A lint gap this work exposes.** No clippy lane in `ci.yml` combines `clt` with `training` or with `diffusion`
   (the lanes are `transformer`, `rwkv,rwkv-tokenizer`, `stoicheia`, `clt,transformer`, `quantized,transformer`,
   `sae,transformer`, `mmap,transformer`, `diffusion`, `diffusion,fused-attn`, `memory`, `memory-debug`,
   `training`). So `tests/validate_clt_train.rs` (`clt` + `training`) and `tests/validate_clt_replacement.rs`
   (`clt` + `diffusion`) are compiled and run in CI but **never linted**, and the new example (`clt` + `diffusion`
   + `training`) would escape the same way. A new clippy lane for `clt,diffusion,training` closes it for all three.
   This is the same class of defect the repository's `CLAUDE.md` records for a feature-gated target (2026-09-22).
3. **Lane parity.** Every CI command is mirrored in `publish.yml` and `scripts/preflight.ps1`, and
   `scripts/lane-parity.ps1` fails the build when they disagree. Adding a lane means editing three files and
   re-running the parity script; getting one wrong fails `ci.yml`'s `lanes` job.
4. **Honesty of the numbers.** The model is a seeded random-weight `OthelloGpt`, so *what the CLT learns about it
   means nothing*; the example demonstrates the mechanics (the loop, the units, the replacement model, the export
   round trip), and must say so in its own output. It must not print "loss recovered 0.9" as if that were a result.
5. **Semver.** Root re-exports are sticky (adding later is free, removing is a breaking change), so the question of
   re-exporting the training types (below, S3) is decided toward the reversible side.

---

## Decisions to take first (recommendation in bold)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | The model | tiny seeded `OthelloGpt` (`diffusion` feature, as the CLT tests); or a real checkpoint through an environment variable (as `quick_start_othello`) | **Tiny seeded `OthelloGpt`.** It is the only model constructible offline: `OthelloGpt::init(config, &varmap, &device, seed)`. A real-checkpoint mode can be a later second example. |
| D2 | Required features | `clt`, `diffusion`, `training` | **All three**, in `[[example]] required-features`. `training` is for `optim::AdamW`; `diffusion` for `OthelloGpt`. |
| D3 | Does CI run it | build only; or build and run | **Build and run**, with the example asserting its own invariants (see "Self-checks"). Costs seconds; buys "the recipe still works". |
| D4 | A new lane | add `clt,diffusion,training` to clippy, rustdoc, build, and run | **Yes**: one clippy lane (closes the gap in "What is at stake" 2), one rustdoc entry, one run step. Edit `ci.yml`, `publish.yml`, `preflight.ps1` together. The tests already run on this feature set (`ci.yml` step "Tests (CLT train + replacement)"), so the gap is lint and docs, not execution. The slug for `-Only`/`-Skip` is `clt-train`. |
| D5 | Data | seeded pseudo-random move sequences generated inside the example (no `rand` dependency) | **Yes.** State plainly that the moves are not legal games. |
| D6 | Root re-exports (S3) | re-export `TrainableClt`/`TrainableCltConfig` at the crate root, or leave them at `candle_mi::clt::train` | **Leave them**, and note the question in `docs/roadmaps/v0.3.0-api-ergonomics.md` (it already tracks root-re-export asymmetries, around its lines 155 to 166). Adding later is free. |
| D7 | Upstream candidates (S4) | file the two `candle` issues; or leave them in the report | **File them**, directly on `huggingface/candle`, after re-measuring on unpatched 0.11.0 and on 0.11.1 (see S4: the report's numbers came from a patched sibling clone); the user decides when. |
| D8 | `ROADMAP.md` "v0.2.3 entry" (S2) | add one; or do not | **Do not invent one.** That file is a phase plan that never tracked the 0.2.x patch releases (it has no entry for 0.2.1 or 0.2.2 either); v0.2.3's scope lives in `CHANGELOG.md` `[Unreleased]`. Fix the stale source layout only. |

---

## The recipe (what the example does, in order)

Everything here is an existing API; shapes are from the doc comments. **Read each signature's doc before calling it**:
this outline was written from the docs and the two test files, not from a compiled sketch.

`tests/validate_clt_replacement.rs` (lines 25 to 60) is the template for the model and the capture; 
`tests/validate_clt_train.rs` (lines 180 to 204) is the template for backward and the Adam step.

1. **Model.** `OthelloGptConfig::new(12, 6, LAYERS, 2, D, false)`, `VarMap::new()`,
   `OthelloGpt::init(config, &varmap, &Device::Cpu, SEED)`. The tests use `LAYERS = 3`, `D = 8`; the example may use a
   slightly larger `D` so a top-`k` of a few features is not degenerate.
2. **Data.** A seeded sequence generator producing `[batch, seq]` `u32` ids (vocab 12, seq 6). Several hundred
   sequences, so there are a few thousand positions.
3. **Capture.** One forward per batch with a `HookSpec` that captures `HookPoint::ResidMid(l)` (each layer's MLP
   input) and `HookPoint::MlpOut(l)` (its output) for every layer, as the test's `clean_mlp` does for `MlpOut`.
   Stack to `x` and `target`, each `[rows, LAYERS, D]` (`rows = batch * seq`). **Then `.detach()` both.** The model is
   built over a `VarMap`, so its forward is tracked and the captured tensors carry the backbone's graph (measured:
   `x.track_op()` is `true`). Left attached, `loss.backward()` walks the whole backbone: 15 training steps took
   723 ms attached against 239 ms detached (3x), for gradients nobody uses. Detach once, after scaling (step 4).
4. **Scales.** Per-layer scalar scales (`scale_in[l]`, `scale_out[l]`, for example the RMS of each layer's column), then
   train on `x / scale_in`, `target / scale_out`. `ScaledClt` and `export_circuit_tracer` both expect exactly this
   convention (`x / scale_in[l]`, `target / scale_out[l]`).
5. **Fit.** `TrainableCltConfig::new(LAYERS, D, N_FEATURES, TOP_K)`; `TrainableClt::new(&config, SEED, &device)`;
   `AdamW::new(clt.vars(), ParamsAdamW { … })` (copy the field set from `validate_clt_train.rs` line 193 onward);
   then a plain loop of `let loss = clt.loss(&xs, &ts)?; let grads = loss.backward()?; opt.step(&grads)?;`,
   logging the loss every few dozen steps. The loop stays in the example, which is the point of the example.
6. **Evaluate in raw units.** `ScaledClt::new(&clt, &scale_in, &scale_out)`; `reconstruct_and_fired(&x)` gives the
   reconstruction and the fired mask, from which `fvu_per_layer(&recon, &x_target)` and `dead_fraction(&fired)`;
   then `replacement_logits_for(&model, &ids, &scaled, Splice::Clean | Splice::Replaced | Splice::Ablated)` and
   `cross_entropy_at(&logits, &positions, &targets)` of each, combined with `loss_recovered(clean, replaced, ablated)`.
   Optionally `replacement_logits_where` with a mask of marked positions, to show the confined splice.
   - **Targets:** the model is bidirectional (`causal = false`), so use the input ids themselves at every position, as
     `validate_clt_replacement.rs` does. (The clean model's own argmax from `top_tokens_at` was tried and changes
     nothing material.)
   - **Print the denominator next to `loss_recovered`.** With random weights the MLPs barely matter, so the gap
     `ablated - clean` is tiny and the ratio is ill-conditioned (measured: 0.007 nats, giving 0.927). The line must
     read as a mechanics check, for example "ablated - clean = 0.0072 nats; loss_recovered is a ratio of that and means
     nothing here".
   - **Hold data out.** Fit on one batch of sequences and evaluate FVU, dead fraction and the splice on a second,
     differently seeded batch, or say in the output that the readings are on the training rows. The spike evaluated on
     training rows only.
7. **Export and reload.** `clt.export_circuit_tracer(&dir, &scale_in, &scale_out, "clt_train_tiny")` into a directory
   under `std::env::temp_dir()`, then `let mut local = CrossLayerTranscoder::open_local(&dir)?`,
   `local.load_encoder(layer, &device)?`, and `local.encode(&residual, layer)?` for one `[d_model]` residual (it is per
   layer and per position, needs `&mut`, and returns a sparse list, not a dense vector). This shows a CLT trained here
   is read by the same code as a published one. Remove the directory afterwards.
8. **Say what the numbers are.** Print one line that the weights are random, so the readings demonstrate the
   mechanics and are not findings.

### Self-checks (the example exits non-zero if any fails)

Thresholds are to be **measured on a first run and then set with a wide margin**, not guessed here:
the loss falls substantially from its first value (spike: 1.030 to 0.021 in 300 steps); `Splice::Replaced` and
`Splice::Ablated` differ from the clean logits (spike: max difference 0.045 and 0.147); every metric is finite and
the `ablated - clean` gap is above a floor, so `loss_recovered` is not a ratio of noise; the exported directory
reloads with the trained `(n_layers, d_model, n_features, top_k)`, and `encode` returns at most `top_k` features whose
ids and values match the trained CLT's active set at the same position. Do not assert that `Splice::Clean` equals the
plain forward as a check of the example: `Clean` is defined as that forward, so it is true by construction (the test
file is where it is proven). Keep the example deterministic (seeded, CPU, `F32`); a
loose threshold on a CPU result is fine, an exact float comparison across platforms is not.

---

## Change list (every file the example touches)

| File | Change |
|---|---|
| `examples/clt_train_tiny.rs` | New. First line `// SPDX-License-Identifier: MIT OR Apache-2.0`. Module doc in the style of `quick_start_othello.rs` (what it does, in numbered steps, and the command to run it). Returns `candle_mi::Result<()>`; follow `CONVENTIONS.md` (annotations, no `unwrap` outside tests). |
| `Cargo.toml` | `[[example]] name = "clt_train_tiny"`, `required-features = ["clt", "diffusion", "training"]`. |
| `.github/workflows/ci.yml` | Four additions: clippy (`--all-targets --no-default-features --features "clt,diffusion,training" -- -W clippy::pedantic`), build (`--examples`), a run step (`cargo run --locked --no-default-features --features "clt,diffusion,training" --example clt_train_tiny`), and **the feature set as one more entry in the `Rustdoc (per feature)` loop** (`for f in …`). Put the run step next to the existing "Tests (CLT train + replacement)" step. |
| `.github/workflows/publish.yml` | The same four, verbatim (parity check 1), including the entry in its own `for f in …` rustdoc loop (around lines 220 to 223). |
| `scripts/preflight.ps1` | Add `@{ Id = 'clt-train'; Features = 'clt,diffusion,training' }` to the `$lanes` table. That one entry generates the clippy **and** the rustdoc step for both toolchains, which is why the rustdoc loops in the two workflows must gain the same entry or `lane-parity.ps1` fails (parity check 2). Add the build and run steps with `-Lane 'clt-train'`. The MSRV 1.91 clippy lane runs on the example too. |
| `scripts/lane-parity.ps1` | No edit expected; run it. Only a new *feature* would need a docs.rs or feature-table row, and no feature is added. |
| `examples/README.md` | A row in "Available Examples" (`clt_train_tiny`, `clt` `diffusion` `training`), an entry in the "Example output:" index near the top of the file, and an "Example output" section with the real captured output, as the other examples have. |
| `CHANGELOG.md` | An `### Added` bullet under `[Unreleased]`. |
| `docs/dogfooding-feedbacks/trainable-clt-and-the-candle-costs-of-training-it.md` | Status header (S1). |
| `ROADMAP.md`, `README.md`, `src/lib.rs` | S2 and S3 below. |

**Expect lint findings on first contact.** The new clippy lane lints `tests/validate_clt_train.rs` and
`tests/validate_clt_replacement.rs` with `-W clippy::pedantic` for the first time (no existing lane combined their
features). Budget for fixing what it finds in those two files, on both toolchains, before judging the lane done.

Verification before any push: `./scripts/preflight.ps1` (the authority, ~6 minutes) and, because this adds a CI lane
and an example, `./scripts/preflight.ps1 -Ci` (both toolchains). Also run the example itself and paste its real output
into `examples/README.md`.

---

## The four loose ends

### S1. The report's status header is stale

`docs/dogfooding-feedbacks/trainable-clt-and-the-candle-costs-of-training-it.md` line 1 says "PARTLY IMPLEMENTED on
branch `clt-train` (2026-10-13, unreleased)". The branch merged (PR #4, merge commit `0a789b8`) and the work is on
`main`, still unreleased.

**Do:** rewrite the header to say: merged to `main` through PR #4, unreleased (v0.2.3 pending); §2 to §5 shipped as
`clt::train`; §8 items 1 and 2 shipped as the `clt::replacement` variant (as the header already explains); §8 item 3
(the example) shipped as `clt_train_tiny` once it does; §9's two `candle` fixes remain upstream candidates (S4).
Keep the existing explanation of why the replacement signature differs from the one §8 sketched. Update the status
**again** when the example lands, and when v0.2.3 is tagged.

### S2. `ROADMAP.md`: the source layout is stale; there is no v0.2.3 entry, and there should not be one

`ROADMAP.md` lines 623 to 624 list `src/clt/` as containing only `mod.rs`. The directory has `mod.rs`, `gemmascope.rs`,
`train.rs` and `replacement.rs` (plus a fixture, `gemmascope_test_fixture.yaml`). The `gemmascope.rs` omission predates
this work.

**Do:** update that listing to the four modules, with a one-line description each (`train.rs`: `TrainableClt`, the
sort-based `top_k`, `export_circuit_tracer`; `replacement.rs`: `Reconstruct`, `ScaledClt`, `Splice`, the replacement
logits and the faithfulness readings; `gemmascope.rs`: read from the source file, do not guess).

**Do not** add a "v0.2.3" entry: `ROADMAP.md` is a phase plan (its phases end at `v0.2.0`, then `v0.3.0` onward) and has
no entry for 0.2.1 or 0.2.2; adding one only for 0.2.3 would be an inconsistency. The release scope is in
`CHANGELOG.md`. If the maintainer wants release tracking in the roadmap, that is a separate decision.

**Also seen, out of scope here:** "23 examples" appears at lines 655, 732 and 863 while `examples/` holds 42 `.rs`
sources (48 entries, the rest being `README.md`, `STYLE_GUIDE.md` and four directories); the layout section is likely
stale elsewhere too. Log it; fix only the `clt/` lines in this pass.

### S3. README and crate docs: the `clt` feature row says nothing about training or replacement

`README.md` line 245 and `src/lib.rs` line 53 both read "Cross-Layer Transcoder support". Nothing in the README
mentions training a CLT or the replacement model (its CLT text is about published CLTs, the Figure-13 work and
the circuit-tracer comparison). `TrainableClt` and the replacement items are reachable only by path
(`candle_mi::clt::train`, `candle_mi::clt::replacement`), while `lib.rs` line 405 re-exports
`AttributionEdge, AttributionGraph, CltConfig, CltFeatureId, CrossLayerTranscoder` at the root.

**Do:**
- Reword the `clt` row in **both** tables (README and `lib.rs`) the same way, for example: "Cross-Layer Transcoders:
  load and analyse a published one, train one (`clt::train`, fitted with `optim::AdamW`, which needs `training`), and
  run its closed-loop replacement model (`clt::replacement`)". Lane-parity check 4 compares only the *names* in the two
  tables, so wording is free, but keep them identical.
- Add the example to the README's capability table or examples pointer, wherever the other CLT examples are linked.
- In `src/lib.rs`, extend the `//! ## Training (differentiable backbones)` section (around line 180) with a short CLT
  paragraph and an intra-doc link to `clt::train` and `clt::replacement`. Per `CONVENTIONS.md`, a link to a
  feature-gated item must be plain backtick text, not link syntax, or rustdoc fails in the lanes without the feature;
  `preflight.ps1` runs rustdoc per lane, so it will tell.
- **Root re-exports: decide D6, and recommend leaving them.** The two training types are a coherent sub-API with many
  companions (`Splice`, `Reconstruct`, `ScaledClt`, five metric functions, `CHUNK`), and adding names to the root later is
  free while removing one is breaking. Note the question in `docs/roadmaps/v0.3.0-api-ergonomics.md`.

### S4. Upstream `candle` candidates (§9 of the report): not candle-mi tasks; a decision, then issues

The report names two measured `candle` behaviours: (a) `where` and the comparison kernels have no broadcast-aware
fast path (a stride-0 operand along the last dimension takes the strided path, ~5× slower; this was what made `TopK`
cost 147 ms a step); (b) `Op::Narrow`'s backward concatenates with zero padding to the full shape instead of adding a
slice into the accumulated gradient (the per-layer slice of a stacked block costing a full-size copy). The report leaves
"whether to file them, and through candle-patched or directly" to the maintainer.

**Do, once the user decides:**
1. **Re-measure first.** The report's numbers are from a CUDA run on the maintainer's GPU against a **sibling clone on
   branch `gradstore-accumulate-0.11`**, not a pristine 0.11.0 (see the report's "Measured against" line). A patched
   backward may price `Op::Narrow` differently, so repeat each measurement on unpatched 0.11.0 and on 0.11.1 (which
   changed `candle-core`'s tree, see "Parked" below), and confirm each behaviour is still present before filing. The
   reproduction is `candle-core` alone, so it does not need the 1.91-compatible toolchain constraint that candle-mi's
   own build has.
2. **A minimal reproduction per item**, a few lines of `candle-core` alone (no candle-mi), with the timing it shows.
3. **File on `huggingface/candle`** as an issue (not a PR) per item, in the house style for candle comments: lead with
   the empirical measurement, raw output in fenced blocks, source lines linked at a pinned commit, no editorialising,
   the AI-usage disclosure line. Link the dogfooding report's section by describing it (the report folder is
   gitignored in hf-fetch-model, but candle-mi's `docs/dogfooding-feedbacks/` is tracked and not ignored, checked
   2026-10-31, so a permalink to the report at a pinned commit is possible once it is pushed).
4. Record the issue links in the report's §9 and its status header.

---

## Validation record (2026-10-31)

A throwaway example following the recipe, `OthelloGptConfig::new(12, 6, 3, 2, 16, false)`, 300 sequences of 6 seeded
ids (1800 rows), `TrainableCltConfig::new(3, 16, 64, 4)`, AdamW `lr = 3e-3`, 300 steps, CPU, `dev` profile
(`opt-level = 1`). It compiled first time and ran in 5.2 s total (incremental build 4.5 s). Measured:

| Quantity | Value |
|---|---|
| Per-layer RMS scales (in / out) | about 0.026 / 0.007 |
| Loss, step 0 to step 300 | 1.030 to 0.021 (300 steps in 3.9 s) |
| FVU per layer (on the training rows) | 0.034, 0.022, 0.016 |
| Dead fraction | 0.318 |
| CE clean / replaced / ablated | 2.4792 / 2.4797 / 2.4864 (ln 12 = 2.4849) |
| `loss_recovered` | 0.927, on an `ablated - clean` gap of 0.007 nats |
| Max logit difference, replaced / ablated vs clean | 0.045 / 0.147 |
| `Clean` vs plain forward | 0 (by construction) |
| Active features at one position after reload | 4 = `top_k`, equal to the trained CLT's |
| 15 steps, attached vs detached captures | 723 ms vs 239 ms |

These are a feasibility record, not thresholds: set the self-check margins from the real example's first run.

## Parked: `candle` 0.11.1 and `zip` 9 (context for the v0.2.3 release)

Not part of this plan's work, but it constrains the release, so a fresh session must know it.

- `candle-core` 0.11.1 (published 2026-10-31) requires `zip ^9.1.3`, which would let the three `zip` ignores in
  `deny.toml` go. It **does not compile on Rust 1.91, 1.92 or 1.93** on x86_64 (`__cpuid_count` is called without an
  `unsafe` block, `repack_x86.rs:589`), and builds from 1.94. candle-mi's MSRV is 1.91, so its preflight fails at the
  `[1.91] Clippy` lane on 0.11.1. Everything on stable passed.
- The maintainer chose to stay at MSRV 1.91 and to nudge candle's maintainers. The comment is posted on PR #4076 (the
  fix; checked 2026-10-31: still open, unmerged, no review decision): <https://github.com/huggingface/candle/pull/4076#issuecomment-6544277682>, with
  the toolchain table (fails on 1.91.1, 1.92.0, 1.93.1; builds from 1.94.1; the PR's change fixes all of them with no
  warning on 1.99.x). **Check again around 2026-11-07**: has #4076 merged, is there a `candle-core` 0.11.2? Until a candle release fixes it, candle-mi's `Cargo.lock` stays on 0.11.0.
- The prepared change (candle 0.11.1 and `zip` 9.2.0 in the lockfile, the three `zip` ignores removed from `deny.toml`,
  a CHANGELOG entry) is easy to redo: `cargo update -p candle-core -p candle-nn`, delete the three `zip` ignore
  entries and their comment from `deny.toml`, replace the `zip` bullet in `CHANGELOG.md`'s Security section, run
  `cargo deny check` and `cargo audit` (both clean), then `./scripts/preflight.ps1`.
- A fresh `candle-core = "0.11"` resolution by anyone building candle-mi on 1.91 to 1.93 already picks 0.11.1 and fails,
  whatever lockfile candle-mi ships. That is candle's regression, and it affects the already-published 0.2.2 too.

## State of the repository when this plan was written

Four local, unpushed commits on `main`: `839b10a` (`hf-fetch-model` 0.12.3 → 0.12.4, the TLS floors), `9de11b5`
(lockfile: `quinn-proto`, `webpki-root-certs`, `tokio`), `c7b6f5e` (lockfile: `tempfile`'s `getrandom` edge) and
`7b1fac8` (their CHANGELOG bullets). `./scripts/preflight.ps1` (default tier) passed on the first three. `-Ci` was not
run. Nothing is pushed.

## Out of scope

- A training loop *in the library* (the report's §1 and "Not asked for": the loop, data and schedule stay with the caller).
- `AuxK` dead-latent revival, TF32 matmuls, a fused top-k kernel (the report's "Not asked for").
- An `impl Reconstruct for CrossLayerTranscoder`, which would let the replacement model run on a *published* CLT: the
  report's §8 item 1 argued for it, and it was not built (only `ScaledClt`, over `TrainableClt`, implements the trait).
  It is a design decision of its own and should be taken separately; the example should not pretend it exists.
- Raising MSRV, the `candle` 0.11.1 move, and the other stale spots in `ROADMAP.md`.
