# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes. Every annotation pattern, doc-comment rule, and style rule in that file is mandatory.

Every `.rs` file must start with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first line.

## Version Control

- **Commit directly to the current branch (normally `main`). Do NOT create a new branch before committing.** This explicitly overrides the default "if on the default branch, branch first" behavior — in this repo, committing onto `main` is correct and expected. Create a branch only if the user explicitly asks for one (e.g. for a PR).
- Commit and push **only when the user asks** — never preemptively. (This part of the default is correct; keep it.)

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo update -p hf-fetch-model` — pick up the latest compatible patch release
2. `cargo fmt`
3. `cargo clippy --all-targets --no-default-features --features <lane> -- -D warnings` for each feature lane the change touches. **Never `--all-features`** — see below.
4. `cargo test`
5. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

CI runs clippy separately for each backend feature. Before pushing, also run clippy with each feature flag individually:
- `cargo clippy --features transformer -- -W clippy::pedantic`
- `cargo clippy --features rwkv -- -W clippy::pedantic`

**`--all-features` is not an option here, for three independent reasons.** It does not run at all on this machine: it enables `metal`, which pulls `objc2`, which `compile_error!`s on anything that is not Apple. It cannot see a lint that fires only under one feature. And it structurally cannot see two whole classes of breakage that the per-lane matrix does catch: a feature-gated intra-doc link (which resolves when its feature is on and dangles when it is off, so `preflight.ps1` runs rustdoc per lane), and a target that compiles only when a feature is *present* — an integration test missing its `[[test]] required-features` entry in `Cargo.toml` builds fine under `--features transformer` and fails in every other lane. Both were live defects caught by preflight and by nothing else (2026-09-22).

When in doubt, do not guess the lane: `./scripts/preflight.ps1` runs the whole matrix in ~6 minutes, CPU only, and is the authority.

Before every push, run `./scripts/preflight.ps1`. It freshens the toolchains (`rustup update stable`, and ensures the MSRV `1.91` toolchain) so local lints match CI's rolling stable — a dry-run on a stale compiler can pass while CI fails on a newer lint (this is how `clippy::suboptimal_flops` from Rust 1.96 broke a clean `main`; the same lint, on the then-MSRV 1.88, also broke a push when preflight didn't yet run the MSRV lane).

Preflight is **tiered** — CI runs a `1.91` (MSRV) + `stable` matrix, and the default fast path does not fully mirror both:
- **Default** (`./scripts/preflight.ps1`): full **stable** mirror (every CI lane on stable) **plus** MSRV `1.91` fmt + clippy. The MSRV clippy lanes catch version-specific lints/compile errors (the gap that bit us) cheaply.
- **`-Ci`** (`./scripts/preflight.ps1 -Ci`): the **full** both-toolchain mirror — every CI step on **both** `1.91` and `stable`. This is the run that literally means "green preflight = green CI"; use it before important pushes and after any MSRV-sensitive change.
- **`-Full`**: also runs the `bench_hook_*` CPU benches (composes with `-Ci`). These **skip on CI** (the gated `Llama-3.2-1B` isn't cached on runners), so they are not part of "green CI"; run `-Full` only when adding a new model family — the change that can shift the benchmarked forward/hook paths.

## Releasing

Cutting a release (a `vMAJOR.MINOR.PATCH` tag → crates.io publish):

**The order matters and is not negotiable: verify, then bump, then dry-run.** `cargo publish --dry-run` contacts the registry, so running it before the bump only proves the *already-published* version packages. It has to run against the version that will actually ship. The expanded checklist is [`docs/roadmaps/release-sequence.md`](docs/roadmaps/release-sequence.md); it follows this same order and must be kept in step with it.

1. **Verify green first, before touching any version string.** Re-run `./scripts/resurrect.ps1` if any forward/CLT/SAE/quantized numeric path changed (GPU, local only — CI never runs the oracles). Then `./scripts/preflight.ps1 -Ci` for the full both-toolchain mirror. Both are version-agnostic, which is why they come first. If the change could plausibly move the benchmarked forward/hook paths (new model family, `HookCache`/`HookSpec` refactor), also refresh [`RELEASE_TIMINGS.md`](RELEASE_TIMINGS.md) with a `--release` run of `bench_hook_overhead`/`bench_hook_diagnostic` — see that file for the exact commands; `preflight.ps1 -Full` runs the same tests but in `dev` profile, so it does not produce a comparable number.
2. **Bump, in one commit.** `version` in **both** `Cargo.toml` and `Cargo.lock` (`publish.yml` fails on a dirty lockfile), promote the CHANGELOG `[Unreleased]` section to `## [X.Y.Z] - <date>` with a fresh empty `[Unreleased]` above it, and bump the README version banner.
3. **Then `cargo publish --dry-run`**, against the bumped version. This is the non-skippable gate that catches what no lane sees: `[package] exclude` rules, metadata completeness, licence headers, the simulated upload.
4. **Push, wait for green CI, then tag.** The tag is what fires `publish.yml` → crates.io (irreversible) — tag only after CI is green on the release commit. Real release tags are `vMAJOR.MINOR.PATCH` only; hyphenated tags (`v0.1.9-plt`) are milestones that do NOT publish.
5. **Cut the GitHub Release — AFTER the crates.io Publish workflow is green**, never before (the crate must actually be live). Use a hand-authored narrative body (title + "In the crate" / "Experiments" / "Verified before tagging" — the v0.1.18/v0.1.19 house style), NOT a raw changelog dump. `scripts/release-notes.ps1 -Version X.Y.Z -Theme "..."` scaffolds it from the CHANGELOG section; edit into prose, then `gh release create vX.Y.Z --title "..." --notes-file <f> --verify-tag --latest`. GitHub Releases are for real `vMAJOR.MINOR.PATCH` tags only (mirrors the publish trigger).

## Oracle-suite resurrection

Every oracle/parity test is `#[ignore]`d (needs cached — often gated — models, and many need a ≥16 GiB GPU), so CI never runs them and "green CI" says nothing about numeric parity. `scripts/resurrect.ps1` runs that suite **locally** (where the models are cached) and stamps `RESURRECTION.md` with a per-test "last verified" record. Tiers: `-Quick` (~6 min ungated smoke: CPU encoder-parity plus the GPU `patch_at` guard, which downloads nothing), default (~40–50 min, all but the Mistral-7B CPU forward + anacrousis), `-Full` (~1.5–3 h, everything). Run it periodically and after any change to a forward/CLT/SAE/quantized numeric path; commit the refreshed `RESURRECTION.md`.

Selecting entries, so a single refresh costs a minute rather than the full tier:
- `-List` prints the number/slug map plus each entry's tier and last-verified date. Start here.
- `-Only <tokens>` / `-Skip <tokens>` take 1-based numbers **or** stable slugs (`-Only longrope`, `-Only 12`, `-Only clt,sae`, `-Skip longrope`). Prefer slugs in anything you write down: inserting an entry renumbers everything after it.
- `-Skip longrope` is the common shortcut: that one entry is ~15 min of the ~44 min default tier (Phi-3.5-mini at F32 spills ~8.8 GiB), so skipping it gives ~28 min whenever the VRAM-spill path is not what changed.
- A partial run stamps `partial (N of <all entries>: …)`, never a tier name, so a three-entry run can never later read as full coverage.

**A default-off feature that no entry enables cannot invalidate the suite.** `training` gates `src/optim.rs` entirely and appears nowhere in `resurrect.ps1`, so adding it after a green run needs no re-run. Check the same way: grep the feature name in `resurrect.ps1` and confirm the module is fully `cfg`-gated.

## Shell Environment

The user runs PowerShell on Windows. Use PowerShell syntax for all suggested commands:
- Use `$env:VAR="value";` instead of `VAR=value` for environment variables
- Use semicolons to chain commands, not `&&`
- Use forward slashes in paths when running Rust/cargo commands
