# SAE Manifold Dilution — Does It Reproduce Off Llama? — Experiment Plan

## Status (2026-10-08)

Design phase. No code written yet. This document is the pre-flight plan; it will be
reviewed before any implementation begins.

| Step | Status | Artefact (planned) |
|---|---|---|
| 0 — Read the released code; establish what the original actually ran | ⏳ Not started | `docs/experiments/sae-manifold-dilution/upstream-audit.md` |
| A — Port the two evaluations to Gemma 2 2B + Gemma Scope | ⏳ Not started | `sae_manifold_dilution` example + `gemma2-2b-16k.json` |
| B — Width sweep inside one SAE family | ⏳ Not started | per-width JSON + `findings.md` capture table |
| C — Metric-sensitivity sweep over the five similarity metrics | ⏳ Not started | per-metric JSON + `findings.md` clustering table |
| D — Readout + figures + friction log + write-up + ship | ⏳ Not started | `findings.md`, `*.wl` + `plots/*.png`, README row, `CHANGELOG.md` |

**Result in one sentence (to be filled at Step D).** _TBD — whether dilution is a
property of SAEs, of a model family, or of the dictionary's architecture and width._

---

## Context

Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, Boppana, Kowal, Shyam, Merullo, Geiger and
Lubana, *Do Sparse Autoencoders Capture Concept Manifolds?*
([arXiv:2604.28119](https://arxiv.org/abs/2604.28119), submitted 30 April 2026, **v1 only,
no revisions**; blog version in Goodfire's Neural Geometry series, 21 May 2026).

The claim. Concepts are often organised along low-dimensional manifolds rather than as
independent linear directions. An SAE can capture such a manifold two ways: **globally**, by
allocating a compact group of atoms whose linear span contains the whole manifold, or
**locally**, by distributing it across features that each tile a restricted region.
Empirically the authors find SAEs do neither cleanly, mixing the two in a fragmented regime
they name **dilution**: manifolds are not captured as coherent subspaces but tiled across
many localised, partially redundant features.

**What has happened since** (literature state as of 2026-10-07):

| Work | Date | Relation |
|---|---|---|
| Subspace-Aware SAEs (SASA), Dalili & Mahdavi, [arXiv:2606.06333](https://arxiv.org/abs/2606.06333) | 4 Jun 2026 | Independent group. Architecture fix: learned decoder subspaces, Top-s group gating, nuclear-norm rank adaptation. GPT-2, Mistral-7B. |
| A Geometric View for Understanding Concept Learning in SAEs, [arXiv:2606.07007](https://arxiv.org/abs/2606.07007) | Jun 2026 | Cites, extends conceptually |
| Structuring Sparsity: Block-Sparse Featurizers, Fel et al., [arXiv:2606.25234](https://arxiv.org/html/2606.25234v1) | Jun 2026 | The authors' own follow-on, vision models |
| Sparse Autoencoders Encode Both Concepts and Functions, [arXiv:2607.24645](https://arxiv.org/pdf/2607.24645) | Jul 2026 | Cites the global/local distinction |
| From Geometric Recovery to Causal Validation, Bal, [arXiv:2607.12166](https://arxiv.org/abs/2607.12166) | 13 Jul 2026 | Adjacent. Does **not** engage dilution. Reproduces Elhage 2022 / Gao 2024; finds up to 77% of geometrically-recovered features causally inert |

**The gap.** Every one of these either assumes the finding and proposes a better
architecture, or works an adjacent question. **Nobody has re-run the evaluation on a
different model family with a different SAE suite to see whether dilution reproduces.**

That is the experiment below. Note the asymmetry it exploits: if dilution is real and
general, a second family confirms it and the original is strengthened. If it does not
reproduce, the interesting question becomes *which* axis carries it, and this plan is
designed to tell those axes apart rather than to produce a verdict.

---

## The hard constraint: the original's model does not fit

Same wall as `PLAN-GEOMETRIC-CALCULATOR.md`:

- `sae-manifold` defaults to **Llama-3.1-8B**. At F32 that is ≈ 32 GB of weights and does
  not fit the RTX 5060 Ti (16 GB). At BF16 it fits weights only, with no headroom for
  activations and hooks, and violates candle-mi's F32-everywhere parity standard.
- Therefore this is a **cross-family method replication**, not a bit-exact reproduction of
  the original's numbers. Layer indices, feature counts and absolute capture scores will
  differ by design. **That difference is the experiment**, not a defect in it.

The repository anticipates this: model and layer are **configurable in `data.py`**, so the
port is a configuration change rather than a rewrite.

---

## Pilot model and SAE ladder

| Role | Model | SAE suite | Status in candle-mi |
|---|---|---|---|
| Anchor | Gemma 2 2B | `gemma-scope-2b-pt-res`, 16K | `sae (encode/inject/parity)` — ✅ PASS, verified 2026-09-21 |
| Width axis | Gemma 2 2B | Gemma Scope, additional widths | to confirm in cache at Step 0 |
| Family axis (post-pilot) | Llama 3.2 1B | whichever open SAE suite is available | not yet surveyed |

**The second axis, and why it matters here.** Gemma Scope is **JumpReLU**. The SAE suite the
original ran on Llama-3.1-8B is **not recorded in this document and must be read off the
released code in Step 0** — do not assume it. If the two differ in activation function or in
sparsity mechanism, then "model family" and "SAE architecture" are confounded in any
cross-family comparison, and the design must separate them before any claim is made.

This is the same structure as the finding already published in the Figure-13 work: redirect
magnitude there ranged 3.8x to 9,974,880x **across transcoders on the same models**, so the
effect belonged to the instrument rather than to the model. Dilution may or may not behave
the same way. The point of Steps B and C is that the plan can tell.

---

## Library prerequisites (Step 0 — infrastructure, not science)

1. **Upstream audit.** Clone [`goodfire-ai/sae-manifold`](https://github.com/goodfire-ai/sae-manifold)
   (MIT; **2 commits, 27 stars, 8 forks** as observed 2026-10-07 — a code drop, not a
   maintained artefact). Record, in `upstream-audit.md`:
   - which SAE suite, width and layer the paper actually used;
   - the exact definition of the subspace-capture score and of the dilution regime;
   - the five similarity metrics (decoder cosine, co-activation, correlation, mutual
     information, inverse Ising couplings) and the community-detection settings
     (Leiden / spectral) as implemented, not as described in the prose;
   - the eight packaged manifolds (colors, temperature, age, geography, days, years,
     formality, sent_length) and how their probe sets are built.
2. **Fork survey.** ✅ **Done 2026-10-08 — see Appendix A.** Six of eight forks are
   untouched; the two that carry real work both stay on Llama 3.1 8B. The cross-family cell
   is still empty.
3. **Ask upstream.** Establish whether a cross-family re-run is known to the authors; a
   negative answer from them is itself citable. Channels ranked in **Appendix B**.
4. **Confirm the Gemma Scope widths present in the local cache**, by reading
   `~/.cache/huggingface/hub`, not by assumption.

**Gate:** do not enter Step A until the subspace-capture score is reimplemented and agrees
with the upstream implementation on at least one of the eight packaged manifolds, to the
usual parity bar. Without that, a cross-family difference cannot be distinguished from a
reimplementation error.

---

## Folder structure

```
docs/experiments/sae-manifold-dilution/
  upstream-audit.md          # Step 0: what the original actually ran
  findings.md                # the write-up
  friction.md                # what broke in a 2-commit repository, and how
  gemma2-2b-16k.json         # Step A
  width-sweep/*.json         # Step B
  metric-sweep/*.json        # Step C
  plots/*.png
examples/
  sae_manifold_dilution.rs
  results/sae_manifold_dilution/
    sae_manifold_plot.wl
```

---

## Experiment

**Step A — Port and anchor.** Run both evaluations (subspace capture; unsupervised feature
clustering) on Gemma 2 2B with Gemma Scope 16K, over all eight packaged manifolds. Report
per-manifold capture scores and the global/local/diluted classification, against the
original's Llama numbers side by side, with the caveat that absolute values are not
comparable across families.

**Step B — Width sweep inside one family.** Repeat Step A across the Gemma Scope widths
present in the cache, holding model, layer and metric fixed. This isolates **dictionary
width** as an axis. If dilution is monotone in width, that is a different claim from the
paper's.

**Step C — Metric sensitivity.** Repeat the clustering evaluation across all five similarity
metrics, holding model, SAE and width fixed. Report the pairwise agreement between the
induced clusterings. This isolates **the measuring instrument** from the thing measured.

**Step D — Readout.** Which of {model family, SAE architecture, dictionary width, similarity
metric} accounts for the variance in the dilution verdict, and in what order.

---

## Predictions (preregistration-style, recorded before any run)

Marked before Step A so the result is judged against an explicit prior.

1. **Dilution reproduces qualitatively on Gemma 2 2B.** Manifolds will be tiled across
   partially redundant features rather than spanned by a compact atom group. Confidence:
   moderate-to-high. The finding is about SAE training objectives, which do not change
   across families.
2. **Absolute capture scores will not transfer.** Confidence: high, and this is nearly
   definitional given the family and width change. Any write-up that compares them directly
   is making an error.
3. **Width will matter more than family.** Dilution will be monotone, or close to monotone,
   in dictionary width, and the width effect will exceed the Gemma-versus-Llama effect.
   Confidence: moderate. This is the prediction most likely to be wrong, and the most
   informative if it is.
4. **The five similarity metrics will not agree with one another.** Pairwise agreement
   between the induced clusterings will be lower than the agreement between any one metric
   across models. If so, the clustering result belongs to the metric rather than to the
   model, which is the same shape as the Figure-13 transcoder finding and would be the most
   useful thing this replication produces. Confidence: moderate.
5. **The eight manifolds will not behave alike.** The cyclic ones (days, years) will be
   captured more globally than the ordinal or continuous ones (age, temperature,
   sent_length). Confidence: low-to-moderate; recorded because it is cheap to check and
   nobody has reported it.

**What a negative result means.** If dilution does not reproduce, the claim is **not**
refuted: a single cross-family cell cannot do that, and this plan reserves "refute" for
nothing. The result would be a boundary condition, and Steps B and C exist to say which
boundary.

---

## Figures (deliverable and visual validation)

1. Per-manifold capture score, Gemma anchor against the original's published Llama values,
   with the non-comparability stated in the caption rather than in a footnote.
2. Capture score against dictionary width, one line per manifold (Step B).
3. Pairwise agreement matrix between the five similarity metrics (Step C). This is the
   figure most likely to be the paper.
4. A projection of one manifold's features, global case against diluted case, as the visual
   check that the classifier is classifying what it claims to.

---

## Deliverables

- `findings.md` with the four tables and the Step D readout.
- `friction.md` — what broke running a 2-commit research repository from outside, and how it
  was resolved. The Figure-13 work's reproducibility lessons came from exactly this kind of
  log, and they were cited by reviewers as a contribution in their own right.
- `examples/sae_manifold_dilution.rs`, runnable end to end on 16 GB.
- A row in the README **Paper replications** table, at the density of the Figure-13 and Maar
  rows, not the one-clause density.
- An upstream note to the authors with the result, whichever way it comes out.

---

## Reverse-engineering provenance

New tensor capture is **not** introduced: the SAE encode/inject path already carries a
Python-vs-Rust parity oracle. The validation gate here is **scoring-function agreement**
(Step 0, gate above), because the entire experiment is a comparison of a score across
conditions, and an unvalidated score makes every comparison meaningless.

| Phase | Purpose | Artefact |
|---|---|---|
| 1. SAE path parity | Already done | `sae (encode/inject/parity)`, gemma-2-2b + gemma-scope-2b-pt-res, ✅ PASS 2026-09-21 |
| 2. Score reimplementation parity | Subspace-capture score agrees with upstream on ≥ 1 packaged manifold | `upstream-audit.md` |
| 3. Clustering parity | Leiden/spectral settings reproduced, one metric, one manifold | `upstream-audit.md` |

---

## Scope discipline — what this document does NOT cover

- **Training new SAEs.** This plan uses existing open suites only. Dictionary training is a
  separate roadmap (see the v0.2.0 dictionary-training scope and the CLT-training brief).
- **Proposing an architecture fix.** SASA and the block-sparse featurizers already occupy
  that ground. This plan measures; it does not repair.
- **Vision models.** The authors' own follow-on covers that and candle-mi has no vision
  backend.
- **Causal validation of the recovered features.** Whether diluted features are causally
  inert is a real and adjacent question (cf. Bal, arXiv:2607.12166) and a natural follow-on,
  but mixing a geometric claim and a causal claim in one study is how the Figure-13 work
  nearly went wrong.
- **A verdict on whether SAEs are the right instrument.** Out of scope, and not answerable
  from one family.

---

## References

- Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, Boppana, Kowal, Shyam, Merullo, Geiger,
  Lubana. *Do Sparse Autoencoders Capture Concept Manifolds?* arXiv:2604.28119, 30 Apr 2026.
- Code: <https://github.com/goodfire-ai/sae-manifold> (MIT).
- Goodfire, *The Neural Geometry Series*, May–July 2026.
- Lieberum et al. *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.*
  BlackboxNLP 2024, arXiv:2408.05147.
- Jacopin. *Planning or Improvisation? Stress-Testing the Poetry Planning Site on Open Models
  and Open Cross-Layer Transcoders.* arXiv:2609.18440 — for the instrument-versus-model
  argument reused in Steps B and C.

---

## Appendix A — Fork survey of `goodfire-ai/sae-manifold`

Observed **2026-10-08** via the GitHub API, comparing each fork's default branch against the
parent's `main`. Parent at that date: 27 stars, 8 forks, 2 commits, `has_issues: true`,
0 open issues, last push 2026-05-21.

| Fork | Ahead | Window | What is in the commits |
|---|--:|---|---|
| `ovindu-a` | **13** | 2026-05-29 → 06-24 | SAE *training* files, additional SAE modes, a **subspace SAE**, a literature review, a webapp, a PCA manifold view, and a large committed cache of `subspace_capture` PDFs. **`data.py` unchanged.** |
| `tony10101105` | **3** | 2026-07-29, 8 minutes apart | `ablation_test.py` and ablation result JSONs. Cache directory `llama31_8b_it_layer16`: the Llama 3.1 8B **instruct** variant at **layer 16**, against the paper's base checkpoint and layer 18. |
| `alirezaghl` | 0 | — | identical to parent |
| `nathanhu0` | 0 | — | identical to parent |
| `hsusul` | 0 | — | never pushed |
| `udapy` | 0 | — | never pushed |
| `danielchang2002` | 0 | — | never pushed |
| `razor-ai` | 0 | — | never pushed |

**Verdict.** Neither active fork changes model family. `ovindu-a` extends the method on the
original model; `tony10101105` varies the checkpoint and layer *within* Llama 3.1 8B and has
published nothing. This matches the pattern in the citing literature exactly: extend or
repair the method, stay on Llama. **The cross-family cell is empty.**

**Two observations worth carrying into the design.**

1. `ovindu-a` committed a subspace SAE on **2026-06-05**, one day after SASA appeared on
   arXiv. Almost certainly independent convergence rather than a port. Two people reached
   for "fix the architecture" in the same week, which is evidence that the repair reflex is
   crowded and the measurement question is not.
2. `tony10101105` ran ablations and wrote nothing up anywhere public. That is precisely the
   failure `friction.md` exists to avoid in this plan.

**Staleness.** These counts are a snapshot. Re-run the survey before Step A if more than a
month has passed:

```sh
gh api repos/goodfire-ai/sae-manifold/forks --paginate \
  --jq '.[] | "\(.full_name) | pushed \(.pushed_at)"'
gh api "repos/goodfire-ai/sae-manifold/compare/main...<USER>:main" \
  --jq '{ahead:.ahead_by, status, commits:[.commits[].commit.message]}'
```

---

## Appendix B — Where to ask whether a cross-family re-run exists

Ranked by expected value, not by convenience. A negative answer is the point: it is what
makes the gap citable rather than merely unobserved.

| # | Channel | Why | Cost |
|---|---|---|---|
| 1 | **In person at BlackboxNLP**, Budapest, 29 Oct 2026 | A co-author of the paper is a keynote speaker there, and I am presenting in the same workshop. Best possible source, and a conversation rather than a cold request. | One conversation |
| 2 | **The mechanistic-interpretability Discord** | Where the Goodfire series surfaced in the first place, and the one venue where a private or unpublished re-run would be mentioned. Not indexed by search, so it cannot be ruled out from outside. | One message |
| 3 | **A GitHub issue on `goodfire-ai/sae-manifold`** | Issues are enabled and there are **zero open**, so a question is visible rather than buried. Creates a public, dated record of the answer, which has value if this becomes a paper. Low probability of reply on a 2-commit code drop, but non-zero. | Minutes |
| 4 | **Email the corresponding author** | Only if 1–3 go quiet. | Minutes |

**If the issue is opened: three or four lines, no more.** State what is being asked, why, and
what has already been checked (the fork survey). The `transformers` #49051 thread is the
lesson: the analysis earned the engagement, the length cost goodwill. Put the question first
and the evidence behind it.
