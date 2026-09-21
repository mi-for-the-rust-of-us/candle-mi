# candle-mi release sequence

Canonical checklist for a `vX.Y.Z` release on crates.io. Distilled from the
v0.1.9 release prep (2026-04-19). Replaces any ad-hoc derivation of the
release steps on future version bumps (v0.1.10, v0.1.11, …).

## When to use this doc

On a commit that is ready to become a release tag. Before starting:

- [`CLAUDE.md`](../../CLAUDE.md) pre-commit gate has already passed
  (`cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test`).
- `CHANGELOG.md` has bullets under `[Unreleased]` for every user-visible
  change since the last tag.
- All feature work for this release has landed on `main`.
- The next release artefact (e.g. `findings.md`, README rows) is committed.

## The sequence

> **Order is load-bearing: verify, then bump, then dry-run.** This mirrors
> [`CLAUDE.md`](../../CLAUDE.md) §Releasing exactly; if the two ever diverge,
> CLAUDE.md wins and this file is the bug. The reason the dry-run comes *after*
> the bump is that `cargo publish --dry-run` contacts the registry: run against
> an already-published version it either objects or proves nothing useful. It
> has to see the version that will actually ship.

### 1. Verify green, before touching any version string

Both steps here are version-agnostic, which is exactly why they come first.

- **Oracles, if a numeric path moved.** `./scripts/resurrect.ps1` (default tier,
  GPU, ~40–50 min). Required after any change to a forward / CLT / SAE /
  quantized path — CI never runs the oracle suite, so green CI says nothing
  about numeric parity. Commit the refreshed `RESURRECTION.md`. See
  [`CLAUDE.md`](../../CLAUDE.md) §Oracle-suite resurrection for tiers and
  `-Only`/`-Skip` selection. **Never run it concurrently with preflight.**
- **The full CI mirror.** `./scripts/preflight.ps1 -Ci` — every CI step on
  **both** `1.91` and `stable`. This is the run that means "green preflight =
  green CI".

### 2. Bump version

- `Cargo.toml` → `version = "X.Y.Z"`.
- Refresh `Cargo.lock` so it matches: `cargo update -p candle-mi`.
  Also good release hygiene: `cargo update -p hf-fetch-model` to pick up any
  patch release of the download crate.
- Bump the **README version banner** (the `> **Note:** vX.Y.Z` line near the top).

### 3. Consolidate CHANGELOG

- Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (use today's UTC date).
- Insert a fresh empty `## [Unreleased]` section above it.
- Preserve the existing `### Added` / `### Changed` / `### Fixed` / `### Tests`
  subsections under the new `[X.Y.Z]` header.

### 4. Commit the release bump

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md README.md
git commit -m "chore: bump version to X.Y.Z and update CHANGELOG"
```

**CLAUDE.md rule:** `Cargo.toml` and `Cargo.lock` go in the **same commit**.
`publish.yml` fails on a dirty `Cargo.lock`.

### 5. Dry-run the publish: the non-skippable gate

```bash
cargo publish --no-default-features --features transformer --dry-run
```

This is what catches packaging issues no lane can see: missing files under
`[package] exclude`, metadata validation, licence headers, the simulated
crates.io upload. The standing rule that this step is never skipped is recorded
as [`feedback_dry_run_before_tag.md`](../../../../.claude/projects/c--Users-Eric-JACOPIN-Documents-Code-Source-candle-mi/memory/feedback_dry_run_before_tag.md).

Step 1's `preflight.ps1 -Ci` already covered the lane gauntlet on both
toolchains, so there is no second transcription of the workflow commands to
keep in sync here. That is deliberate: see [Drift check](#drift-check).

### 6. Push; wait for remote CI green

```bash
git push
```

Wait for both matrix entries (MSRV 1.91 and Stable) to report green in the
GitHub Actions UI. Do not proceed until both are ✓.

### 7. Tag and push

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

`publish.yml` fires on the tag match (`tags: ["v*", "!v*-*"]` — hyphenated
tags like `v0.1.9-plt` do **not** publish; they are git-level milestones).
The workflow runs the full lane gauntlet again on Ubuntu, then the real
`cargo publish` step — publishes to crates.io.

Monitor the workflow run. On success, the crate is live at
`https://crates.io/crates/candle-mi/X.Y.Z`.

### 8. Cut the GitHub Release, after the crates.io publish is green

Never before: the crate must actually be live first. Use a hand-authored
narrative body (title, then "In the crate" / "Experiments" / "Verified before
tagging" — the v0.1.18/v0.1.19 house style), **not** a raw changelog dump.

```bash
./scripts/release-notes.ps1 -Version X.Y.Z -Theme "..."   # scaffolds from the CHANGELOG
# edit the scaffold into prose, then:
gh release create vX.Y.Z --title "..." --notes-file <f> --verify-tag --latest
```

GitHub Releases are for real `vMAJOR.MINOR.PATCH` tags only, mirroring the
publish trigger.

## Why each gate exists

- **Step 1, `resurrect.ps1`**, is the only thing that says anything about
  numeric parity. Every oracle test is `#[ignore]`d because it needs cached,
  often gated models and a ≥16 GiB GPU, so CI never runs one. A perfectly green
  CI is compatible with a silently wrong forward pass.
- **Step 1, `preflight.ps1 -Ci`**, is about not pushing a broken release-prep
  commit to `origin` at all. Remote CI is ~10 min of feedback latency, and on
  the release commit specifically — the thing the tag will point at — you do not
  want a "fix CI for vX.Y.Z" commit polluting `git log` right before the tag.
- **Step 5, `cargo publish --dry-run`**, catches what neither of the above can
  see: it builds the package under `[package] exclude` rules, validates metadata
  completeness, verifies licences, and simulates the crates.io upload. Recorded
  as non-skippable in
  [`feedback_dry_run_before_tag.md`](../../../../.claude/projects/c--Users-Eric-JACOPIN-Documents-Code-Source-candle-mi/memory/feedback_dry_run_before_tag.md).

## Drift check

This doc deliberately holds **no transcription of the workflow YAML**. It used
to: steps 4 and 6 were hand-copied `cargo` chains mirroring `ci.yml` and
`publish.yml`, and they had to be re-diffed against the workflows before every
use because they rotted silently. `preflight.ps1` is now the maintained
instrument for that job, and `-Ci` runs every CI step on both toolchains, so
the only correct move is to call it rather than re-copy it.

The one drift risk that remains is **`preflight.ps1` itself falling behind the
workflows**. Check that, not this file:

```bash
grep -E "^\s+run: cargo" .github/workflows/ci.yml
grep -E "^\s+run: cargo" .github/workflows/publish.yml
```

If a lane appears there that `scripts/preflight.ps1` does not run, fix the
script.

## Alternative: `act`

[`nektos/act`](https://github.com/nektos/act) runs `.github/workflows/*.yml`
literally in Docker, using the same `ubuntu-latest` image as GitHub Actions.
Highest-fidelity dry-run possible: it catches Linux-specific issues that any
Windows-local run misses. Downsides: requires Docker Desktop, has quirks with
`Swatinem/rust-cache@v2` and some GitHub-context-dependent actions. Optional.
`preflight.ps1 -Ci` catches the large majority of issues without it, and the
Linux-only residue is what step 6's remote CI is for.

## Cross-reference

- Pre-commit hygiene: [`CLAUDE.md`](../../CLAUDE.md) §Pre-commit Checks.
- Publish trigger rules (which tag names publish vs git-only milestones):
  [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml)
  header comment. Tags `v0.1.9-plt`, `v1.0.0-rc.1`, etc. are excluded by the
  `!v*-*` filter.
- Dry-run rule origin: `feedback_dry_run_before_tag.md` (user memory).
