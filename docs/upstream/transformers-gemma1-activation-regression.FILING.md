# FILING SHEET: `huggingface/transformers` bug report

Companion to [`transformers-gemma1-activation-regression.md`](transformers-gemma1-activation-regression.md),
which stays the full document. This one is the same material cut into the fields of
`.github/ISSUE_TEMPLATE/bug-report.yml`, in the form's own order, so it can be pasted
straight into the web form without re-editing mid-filing.

**File it at:** https://github.com/huggingface/transformers/issues/new?template=bug-report.yml

The form is a YAML form with structured fields, so `gh issue create --body-file` would
not populate them. Use the web form.

---

## TITLE

```
Gemma 1 checkpoints silently use exact GELU instead of gelu_pytorch_tanh since v4.48.0 (#35235 removed the hidden_activation guard)
```

Leave `#35235` bare: GitHub auto-links issue numbers in titles, and a markdown link
would render literally.

---

## FIELD 1: System Info

```
- transformers 5.1.0, torch 2.10.0+cu130, Python 3.14, Windows 11
- Version bisect on the guard, by release tag: present through v4.47.1, gone in v4.48.0
- google/gemma-2b at revision 9cf48e52b224239de00d483ec8eb84fb8d0f3a3a
- Cross-checked against an independent Rust implementation (candle-mi), which is how
  this surfaced
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

## FIELD 5: Reproduction

Paste, in this order, from the main document:

1. **Summary**: what happens, the five affected repos, what is unaffected, and the
   workaround. (Lead with the workaround: anyone arriving from a search wants it first.)
2. **Root cause**: the `#35235` diff, `gemma2` keeping its guard in the same commit,
   the history of `#29402`/`#29995`, and `hidden_activation` being removed in v5.0.0 so
   the override is now unreachable.
3. **Reproduction**: the two-line `GemmaMLP(config).act_fn` check, the "no warning is
   emitted" evidence, the eight-repo resolution table, and the derivatives paragraph.
4. **Measured effect**: the three-prompt table, the 30-logit summary, the runnable
   script, and the candle-mi cross-check.
5. **Why this matters, given that the rankings do not move**: the Hub discussion #39
   precedent, the single-token caveat, the train/serve framing.
6. **Why this has gone unnoticed for 21 months**: the four reasons.
7. **Prior art checked**: including the `cc @danielhanchen` line, which belongs here
   rather than in field 2.

---

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

1. Update the tracker row in [`README.md`](README.md): status `DRAFT, not filed` becomes
   `FILED <date>` with the issue number and link, matching the five existing entries.
2. Update the `Status:` line at the top of the main document.
3. If a maintainer picks a fix shape, the PR is small: the mapping plus a regression test
   in `tests/models/gemma/`. Note that `transformers` is modular, so the edit may need to
   go in `modular_gemma.py` and be regenerated; check before writing.
