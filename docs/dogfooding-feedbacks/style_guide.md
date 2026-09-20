# How to write a candle-mi dogfooding report

This folder is not a bug tracker and not a wish list. Each file is a **field report from a real
research workload**: something used candle-mi to do mechanistic interpretability in anger, and the
report records what the crate made easy, what it made hard, and what would have made the
measurement trustworthy. That is why these reports have shipped features (`OthelloGpt`, the
`nn_ops` `track_op` dispatch, `Intervention::PatchAt`, `HookPoint: Ord`, `HookCache::captures`)
rather than accumulated a backlog.

Most reports come from a leg of **askesis**, the research monorepo that consumes candle-mi. That is
the point: the reporter and the maintainer are the same person wearing different hats, so the
report is the handoff between them, and it has to survive the months between the two.

This guide is descriptive. Every rule below is already followed by reports in this folder, and each
names the file it is drawn from. Follow it and a new report will read like a sibling.

## The shape

### Title: state the finding as a claim

The title is an assertion about the crate, not a subject line. The settled form is one clause of
fact and one of consequence, joined by `and`, `but`, `so`, or a colon:

> Seeded init promises more than `StdRng` backs
>
> Backbones should be trainable — and today `backward()` says nothing when they are not
>
> `Replace` is whole-tensor only, so activation patching has to splice by hand
>
> Training is 5.1× slower than `PyTorch`, and the fix candle-mi owns is one `DType` parameter

Use the crate's real identifiers, in backticks. A reader scanning the folder should be able to
reconstruct the crate's history from titles alone, which is why `Feedback on hooks` is not a title
and `The interp API forces stringly-typed hook handling downstream` is.

### Status blockquote: the lifecycle, above the title

A `> **Status: ...**` blockquote sits **above** the `#` title. The reporter writes it when the
report is filed; the crate **edits it** when it responds. This is the field that makes the folder a
record rather than an archive.

    > **Status: ASKED** (2026-09-20), from the canvas leg of askesis. Registration:
    > `askesis/reference/canvas/docs/open-measurements.md`, *"Measurement O"*.

    > **Status: IMPLEMENTED in v0.1.20** (2026-07-27). §7 items 1-3 shipped: the
    > `nn_ops` `track_op` dispatch (§6) ... Deferred per §7: `Dropout`, the lean
    > `logits()` path ...

A resolved status names **the version, the date, what shipped, and what was deferred and why**
(`trainable-backbones.md`, `interp-api-forces-stringly-typed-hook-handling.md`). Where the crate
implemented something *different* from what the report asked for, the status says so and points at
the postscript (`positional-replace-has-no-intervention.md`: "NOT the `Tensor::slice_scatter` the
Ask below recommends"). Never silently ship a variant.

### Metadata block: four fields, in this order

    **Date:** July 26, 2026
    **Source:** askesis `canvas` leg — B1 (the candle training loop; askesis's deferred V9)
    **Affected area:** `src/diffusion/othello.rs`, `src/diffusion/mdlm.rs`, ...
    **Severity:** Feature — with a **silent-wrong-answer** component (§2), which is the part that matters

`Source` names the **leg and the step**, not just the project, because the same crate behaves
differently under a trainer, a probe harness and a paper's figure pipeline. `Affected area` names
files and symbols, never just "candle-mi"; a report aimed at the examples rather than the library
writes `**Affected example:**` instead (`json-output-should-be-self-contained.md`).

The block is not optional. `othello-mdlm-needs-a-carry-channel.md` carries a status blockquote and
then no metadata at all, which is the one place that report departs from the folder.

Add `**Platform:**` when an OS or driver behaviour is load-bearing for even one measurement, and
say which one (`training-throughput-ceiling.md` scopes Windows/WDDM to exactly measurement J2). Add
`**Measured against:**` with crate, `candle`, `rustc` and OS versions whenever the report carries
numbers (`trainable-backbones.md`).

### Severity: say what is *not* wrong, as loudly as what is

Six of the eight reports with a severity line carry an explicit exoneration clause, and this is the
folder's most distinctive habit:

> Papercut — discoverability + feature-gating coherence; **not a blocker** (everything works today)
>
> Ergonomics. **No defect, nothing blocked.** The probe is implementable today.
>
> Durability — **not a defect**, and nothing is blocked.
>
> Throughput ceiling — **not a defect**. The arithmetic is right.

The observed vocabulary is `Papercut`, `Ergonomics`, `Design improvement`, `Feature`, `Durability`,
`Throughput ceiling`. Pair it with the disclaimer, and where a real defect *is* present, point at
the section that holds it rather than escalating the whole report
(`trainable-backbones.md`: "with a **silent-wrong-answer** component (§2), which is the part that
matters").

### Opening: ground the ask in a real use, before asking

The settled opening is a use-case section, not a summary: `## The use case`, `## The use`,
`## The problem`, `## The promise`, or, most explicit about its own purpose,
`## Context, so the asks are judged against a real use`. Say what the workload is, what it had
already established, and what it was trying to do when it hit the crate.

This is the section that earns the ask. A maintainer who knows the measurement at stake can propose
a better API than the one requested, which is what happened in
`positional-replace-has-no-intervention.md`.

### Body: numbered findings, headings that conclude

With more than one finding, number the sections (`## 1.`, `## Finding 1`, `## Item 1`) and make
each heading state its conclusion:

> `## 2. The finding that matters: backward() succeeds and trains 1 parameter in 29`
>
> `## Item 4 — the 11% that no GPU will fix`
>
> `## 5. There is no hook point for the logits, and that lands on the verification control`

**Label every claim MEASURED or HYPOTHESIS**, and make each hypothesis name the measurement that
would settle it. `training-throughput-ceiling.md` states this as a rule in its own text and then
opens with four of the author's own confident beliefs, all four refuted within the hour, in a
table. That table is the most useful thing in the file.

Paste real tool output verbatim in fenced blocks. Never reconstruct it from memory.

### `## Ask` / `## Concrete asks`: numbered, with signatures

Give the proposed API as Rust, not prose, and state the compatibility contract explicitly (the
carry report: `forward(input_ids, hooks) == forward_with_carry(input_ids, None, hooks)` "so every
existing caller, hook, capture and intervention is untouched"). Order by how much each would have
helped, and say which one matters most.

Leave genuinely open choices to the crate, in as many words: "whether this is a new trait method
with a default, or a field on `HookSpec`, is the crate's call". Then the crate's answer is a
decision, not an override.

### `## Not asked for`: bound the scope

A distinctive candle-mi section (`othello-mdlm-needs-a-carry-channel.md`,
`positional-replace-has-no-intervention.md`). List what the report is deliberately *not*
requesting, so the crate can size the change without guessing at hidden scope.

Its sibling is the reporter defending the crate's scope against the reporter's own ask:
`## 7. Scope recommendation: differentiability, not a training loop` and `## What NOT to do`. A
report that says where the crate should stop is worth more than one that only asks.

### Closing sections

| Section | Purpose | Required |
|---|---|---|
| `## Why it is a crate change and not a <consumer> one` | the argument for ownership | when the consumer could plausibly have done it itself |
| `## What worked, and is worth keeping` | what the maintainer must not break | strongly encouraged |
| `## How to reproduce` / `## The method note` | the harness, the commands, the machine | whenever the report carries numbers |

`## What worked` matters more than it looks. `interp-api-forces-stringly-typed-hook-handling.md`
devotes a section to it inside a report that is otherwise six complaints, and it is the only signal
telling the maintainer which parts are load-bearing.

## The crate's side of the loop

Unlike a pure feedback folder, these files are **edited by the crate after the fact**, in three
places. Keeping them current is not optional: a report whose status still reads `ASKED` after the
feature shipped is stale, not historical.

1. **The status blockquote**, updated in place (above).
2. **`## Postscript`**, when the implementation the report recommended turns out to be wrong.
   `positional-replace-has-no-intervention.md` carries
   `## Postscript: the one-call implementation this report recommends is wrong on GPU`, and leaves
   the original ask standing underneath it. **Do not edit the reporter's prose to hide the error.**
   The wrong recommendation and its refutation are both evidence.
3. **`## Closure (<date>)`**, recording what shipped and what was learned on the way, including
   findings that outlived the report ("implemented, and it found a `candle` bug on the way").

A reporter may also append a dated pass to their own report rather than revising it in place. The
carry report's `## Consistency pass (2026-09-20, same day), against the crate as it is` corrects
two claims in its own Ask, including a citation it had invented. Appending beats rewriting, because
the correction is itself a finding.

## The standards that make these reports work

**Calibrate before claim.** The house method: implement against a trusted oracle and a stated
numeric bar before claiming anything. Quote the bar, the measured value, and the null band as three
distinct numbers (`training-throughput-ceiling.md`: worst disagreement `9.5e-8`, candle's own
CPU-vs-GPU null band `2.0e-7`, house pass bar `5e-3` on GPU, "all three numbers are distinct and
are kept distinct below"). The program bars are ~1e-3 CPU and 5e-3 GPU.

**Every claim carries a measurement with units.** Not "training is slow" but "505 ms against 99 ms
for the same step in PyTorch, 5.1x". Numbers make a report checkable years later, which is the only
reason old reports still have value.

**Record your own mistakes, and name them as yours.** "J3 and J4 were mine, argued from reading
candle-mi's source, and I was ready to spend days on both." This is not self-flagellation. The next
reader will make the same inference from the same source, and an inference that plausible is
usually evidence for a documentation or API problem.

**Distinguish "the crate is wrong" from "the crate made me work".** Most entries here are the
second kind, and the severity line is where the distinction is made. `HookCache` not being
enumerable is correct behaviour and still a finding.

**Say what the finding would have cost silently.** The strongest reports name the confound that the
gap would have introduced into a published measurement: probing a copy of the model rather than the
trained model itself, or a hook reading a residual stream the logits did not come from. That is the
argument that moves an interpretability crate, more than convenience ever will.

**Aim a request at a version.** Name the next version in the ask, which forces "is this the next
thing, or the thing after" at writing time.

## Mechanics

- Wrap prose at about 100 columns. Seven of the ten reports sit between 98 and 105; one older
  report runs to 121. Titles, the
  metadata block, table rows and pasted output may exceed it.
- File name: `<subject-in-kebab-case>.md`, naming the **subject**, not the consumer, since one leg
  files many reports. `diakrisis-intervention-dogfood.md` and
  `transluce-ai-assessment-for-candle-mi.md` predate this rule; leave them, do not rename.
- Markdown links between reports are relative file names, so the folder browses offline.
- Reports are written in their reporter's own voice, and em dashes, arrows, `x` for multiplication
  and section marks are used freely. This folder is prose, not a paper.
- Cross-reference the crate's own docs by path when a rule lives there rather than here:
  `docs/adding-a-model.md` holds the porting contract, `CONVENTIONS.md` the code rules.
- Write the report while the evidence is still on screen. Every number in these files was pasted,
  not remembered.

## One thing this folder is not

`transluce-ai-assessment-for-candle-mi.md` is a cooperation assessment of an external lab, not a
dogfooding report, and it follows none of the above. It is here for want of a better home. If a
second document of that kind appears, move both to `docs/assessments/` rather than bending this
guide to cover them.
