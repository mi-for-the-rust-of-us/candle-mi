# FILING SHEET: `huggingface/transformers` bug report

Companion to [`transformers-gemma1-activation-regression.md`](transformers-gemma1-activation-regression.md),
which stays the full document. This one is the same material cut into the fields of
`.github/ISSUE_TEMPLATE/bug-report.yml`, in the form's own order, so it can be pasted
straight into the web form without re-editing mid-filing.

**File it at:** https://github.com/huggingface/transformers/issues/new?template=bug-report.yml

The form is a YAML form with structured fields, so `gh issue create --body-file` would
not populate them. Use the web form, or assemble the body and `PATCH` it (below).

> **Read this before using this sheet as a template for another filing.**
> The first version of it got five fields right and the sixth wrong, and the wrong one
> shipped: FIELD 5 held a *table of contents* ("paste the Summary, then the Root cause,
> then...") while the other five held literal paste-blocks, and nothing marked the
> difference. Under copy-paste the list reads as one more block, so
> [#49051](https://github.com/huggingface/transformers/issues/49051) was filed with
> paste-instructions in place of its evidence and had to be corrected two minutes later.
>
> The lesson is not "be careful". It is that **a sheet must not mix paste-blocks with
> instructions about paste-blocks.** Every field below is now either a literal block or
> explicitly marked as assembled by script.

---

## TITLE

```
Gemma 1 checkpoints silently use exact GELU instead of gelu_pytorch_tanh since v4.48.0 (#35235 removed the hidden_activation guard)
```

Leave `#35235` bare: GitHub auto-links issue numbers in titles, and a markdown link
would render literally.

---

## FIELD 1: System Info

Pulled from the document's `## System info` section, not retyped here.

An earlier version of this sheet held a second copy, and the two drifted within a day:
the sheet's copy gained a candle-mi cross-check bullet that the document never got, and
the version filed named candle-mi without linking it. One source, assembled by the same
script as FIELD 5:

```python
field1 = sections["System info"]
```

---

## FIELD 2: Who can help?

```
@ArthurZucker @Cyrilvallez
```

Two, not three: the template says "Please tag fewer than 3 people". Both are on its
text-models line, and `git blame` puts both on the commit in question (#35235 authored
by @ArthurZucker, merged by @Cyrilvallez), which is the routing method the template
itself recommends.

`@danielhanchen` is **cc'd in the body instead**, not here, so the field stays within
the cap. He diagnosed the original exact-versus-approximate GELU problem in #29402, and
this report is that correction having been undone.

---

## FIELD 3: Information (checkboxes)

- [ ] The official example scripts
- [x] **My own modified scripts**

The reproduction is a short standalone script, not an `examples/` script.

---

## FIELD 4: Tasks (checkboxes)

- [ ] An officially supported task in the `examples` folder
- [x] **My own task or dataset (give details below)**

Exact-logit parity checking against an independent implementation. Worth one line in
the body, since it is the only way this class of bug surfaces.

---

## FIELD 5: Reproduction (ASSEMBLED BY SCRIPT, not pasted by hand)

This field is the whole report, which is too long to assemble by hand without losing a
section. Build it from the main document instead, then paste or `PATCH` the result:

```python
import io, re
doc = io.open("transformers-gemma1-activation-regression.md", encoding="utf-8").read()
parts = re.split(r"^## ", doc, flags=re.M)
sections = {}
for p in parts[1:]:
    name, _, rest = p.partition("
")
    sections[name.strip()] = rest.strip()

want = ["Summary", "Root cause", "Reproduction", "Measured effect",
        "Why this matters, given that the rankings do not move",
        "Why this has gone unnoticed for 21 months", "Prior art checked"]
field5 = "

".join(f"#### {w}

{sections[w]}" for w in want)
```

Two details that matter:

- **Demote the headings to `####`.** The form renders its own field labels as `###`, so
  the document's `##` would outrank them and the body would read as if the fields were
  subsections of the content.
- **`Suggested fix` and `System info` are deliberately excluded** from this field: they
  are fields 6 and 1. `Prior art checked` *is* included, because it carries the
  `cc @danielhanchen` line.

If the issue is already open, patching beats re-filing and does not re-notify the people
already tagged:

```sh
gh api repos/OWNER/REPO/issues/N --method PATCH -F "body=@<absolute-path>"
```

Note the path: Python on Windows resolves `/tmp` to `C:\tmp`, which is not Git Bash's
`/tmp`, and `gh` will not find a file written to one and read from the other. Use an
explicit absolute path for both.

## FIELD 6: Expected behavior

```
Loading any Gemma 1.0 checkpoint should apply the tanh approximation of GELU
(gelu_pytorch_tanh), the activation these models were trained with, as transformers did
through v4.47.1 and as GemmaConfig.hidden_act still defaults to.

Today the five google/gemma-1.0 repos resolve to GELUActivation (exact erf) because
their config.json carries the legacy "hidden_act": "gelu", and the guard that used to
correct it was removed. Either the correction is restored in the library, or at minimum
the case stops being silent.

Suggested fixes, in order of preference:

1. Restore the legacy mapping in GemmaConfig.__init__: when hidden_act == "gelu", set
   it to "gelu_pytorch_tanh" and warn once. This reaches every affected checkpoint
   including the derivatives, and keeps the correction in one place.
2. Update the five config.json files on the Hub. Correct at the source, but does
   nothing for already-downloaded caches or for the public derivatives that copied the
   legacy value.
3. At minimum, warn when a Gemma 1 config resolves to exact GELU.

Option 1 is the one that reaches the derivatives, which is why it is first; option 2
complements it at the source, and doing both would be ideal.

Happy to open a PR for whichever shape is preferred.
```

---

## AFTER FILING

1. ~~Update the tracker row in [`README.md`](README.md)~~ **done**: `FILED 2026-09-23`
   as [#49051](https://github.com/huggingface/transformers/issues/49051).
2. ~~Update the `Status:` line at the top of the main document~~ **done**.
3. If a maintainer picks a fix shape, the PR is small: the mapping plus a regression test
   in `tests/models/gemma/`. Note that `transformers` is modular, so the edit may need to
   go in `modular_gemma.py` and be regenerated; check before writing.

---

## THE THREAD, after filing

**2026-09-23 14:02** filed as [#49051](https://github.com/huggingface/transformers/issues/49051),
labelled `bug`. Body corrected at 14:04 and 14:09 (see the note at the top of this sheet, and
the System Info de-duplication).

**2026-09-23 ~14:55**, about fifty minutes after filing, [@vasqu](https://github.com/vasqu)
(Collaborator, and on the template's own text-models line) replied:

> Open to have a fix for this 🤗 imo the best is to fix this in the post init of the config
> and warn there. So we restore that logic not on init of the model but post init of the config
> wdyt?

So the fix is wanted, and the shape is theirs: config `__post_init__`, not model init. That is
**better than option 1 as this report proposed it**, for a reason worth keeping: `__post_init__`
runs after `from_dict`, so it covers configs loaded from the Hub, which is the entire affected
population. Constructing a `GemmaConfig` in Python was never where this bites.

### Reply POSTED 2026-09-23 16:31 as [comment-5798653482](https://github.com/huggingface/transformers/issues/49051#issuecomment-5798653482)

```markdown
Agreed, and post-init is the better hook than the `__init__` I suggested: it runs after `from_dict`, so it covers configs loaded from the Hub, which is the whole affected population. Constructing a `GemmaConfig` in Python was never really where this bites.

There is precedent sitting right there, too: `PretrainedConfig.__post_init__` already carries the `torch_dtype` -> `dtype` shim. Its comment notes that one deliberately does not warn, because most Hub configs carry `torch_dtype` so it would fire every time. This case is the mirror image and should warn: `"hidden_act": "gelu"` is the anomaly on a Gemma 1 config, not the norm.

One question before I write it. Should the mapping fire whenever `hidden_act == "gelu"`, or only when the value came from a file? Unconditional is my instinct, since anyone passing `"gelu"` explicitly for Gemma 1 is most likely making the same mistake, but you may want an escape hatch for the deliberate case.

Happy to open the PR: the edit in `modular_gemma.py` with `configuration_gemma.py` regenerated, plus a test that a config carrying the legacy value resolves to the tanh activation. I should get to it tomorrow.
```

Posted with:

```sh
gh api repos/huggingface/transformers/issues/49051/comments -F "body=@<absolute-path>"
```

Next move is theirs: the reply ends on a question (unconditional mapping, or only for
values that came from a file), so the PR shape is not fully settled until they answer.

### 2026-09-24: two PRs arrived, so we verified instead of competing

[#49061](https://github.com/huggingface/transformers/pull/49061) (16:05:20Z) and
[#49063](https://github.com/huggingface/transformers/pull/49063) (16:48:05Z) were both
opened within the hour of our 16:31:22Z reply, so no PR of ours was needed. We posted a
measurement instead, as [comment-5811228231](https://github.com/huggingface/transformers/issues/49051#issuecomment-5811228231).

The full record, including the BF16 finding that falsified our own filed claim and the
50-model modular-conversion failure in #49063, is in
[the report's verification section](transformers-gemma1-activation-regression.md#verification-of-the-two-candidate-prs).
The harness is [`verify_gemma_activation.py`](verify_gemma_activation.py).

**The lesson for the next filing**, which is what this sheet is for: a narrow
`check_modular_conversion --files <one model>` passes where the repo-wide run fails.
When reviewing a change to a class other models inherit, run the repo-wide check, or
the review reports the opposite of the truth.

### 2026-09-24 afternoon: the maintainer's response, and PR [#49084](https://github.com/huggingface/transformers/pull/49084)

[@Cyrilvallez](https://github.com/Cyrilvallez) closed both earlier PRs as
`AI trying to stiff issue of someone else`, confirmed the fix belongs in the config, and
independently confirmed our finding 4 (telling #49063's author that the consistency
failure was caused by the PR, not pre-existing, after that author had claimed the
opposite). He also asked, fairly, that we not post long comments:

> However note that we are all human reviewers here, so please avoid huge AI dumps of
> unreadable text here please

Answered in three lines, acknowledging the length without withdrawing the content, and
the long comment was left standing as the reference it is.

Our PR is [#49084](https://github.com/huggingface/transformers/pull/49084): the remap in
`GemmaConfig.__post_init__` in `modular_gemma.py`, `configuration_gemma.py` regenerated
by the converter rather than hand-edited, and six tests. Gates before pushing, all green:
`ruff==0.14.10` check and format, repo-wide `check_modular_conversion --check_all`,
`config_docstrings`, `config_attributes`, `copies`.

**Mutation-checked, because a green test proves nothing on its own.** With the fix
reverted and the tests kept, the four that assert the correction fail and the two that
pin scope pass, which is the intended shape. End-to-end on `google/gemma-2b`, our branch
gives `config.hidden_act`, `save_pretrained` and `ACT2FN[config.hidden_act]` all
corrected, and logits bit-identical to the reference on CPU/F32 and CUDA/BF16.

**The collision, and the process error behind it.** #49063's author opened a third PR,
[#49081](https://github.com/huggingface/transformers/pull/49081), at 10:26Z, redoing it
in the config as instructed, two hours before we said we would open ours. We did not see
it: the thread was read by listing comments and the two known PR numbers, never by
listing everything cross-referencing the issue. So we promised and opened a PR that
duplicated an existing one.

**Rule, so the next filing cannot repeat it:** before promising or opening a PR on an
issue, list every cross-reference on it, not just the PRs already known:

```sh
gh api repos/OWNER/REPO/issues/N/timeline --paginate \
  -H 'Accept: application/vnd.github.mockingbird-preview+json' \
  --jq '.[] | select(.event=="cross-referenced") | .source.issue.number'
```

On finding it, we said so on our own PR and offered to close ours in its favour. Not
mentioned there, because it would be a pile-on rather than information the maintainers
lack: #49081 also fails repo-wide modular conversion, on gemma's own two files, having
hand-edited the generated file (deleting its `do NOT edit` banner) and left an unused
`logger` in `modeling_gemma.py`. That is a missing regeneration, fixable in one command,
not a design flaw.

### Groundwork for the PR, already checked

Verified 2026-09-23 against the live repository, so tomorrow does not start from scratch:

- **`PretrainedConfig.__post_init__(self, **kwargs)` exists**, at `configuration_utils.py:314`. It
  is already the home for config back-compat shims: the `torch_dtype` -> `dtype` migration lives
  there, with a comment explaining that it deliberately does not warn because most Hub configs
  carry `torch_dtype` and it would fire every time. Our case is the mirror image and should warn.
- **`configuration_gemma.py` is generated** from `modular_gemma.py` and "one of our CI enforces
  this", per its own header. **The edit goes in `modular_gemma.py` and gets regenerated.**
- **`GemmaConfig` has no `__post_init__` today.** In `modular_gemma.py` the class is at line 51
  and `hidden_act: str = "gelu_pytorch_tanh"` at line 90. So the change adds one that calls
  `super().__post_init__(**kwargs)` and then applies the legacy mapping.
- **The pattern is well-worn**: 135 modular model files already define `__post_init__`.

Remaining to do: fork and clone `transformers`, make the edit, regenerate, add a test under
`tests/models/gemma/` asserting that a config carrying `hidden_act="gelu"` resolves to the tanh
activation, and run `make fixup`.
