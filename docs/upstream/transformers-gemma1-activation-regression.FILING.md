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

Note the path: Python on Windows resolves `/tmp` to `C:	mp`, which is not Git Bash's
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
