# Preflight checks — run before every push.
#
# Mirrors the per-feature lanes in .github/workflows/ci.yml so that a clean run
# here predicts a clean run on CI. CI runs a matrix of two toolchains:
# MSRV (1.91) and stable, each executing fmt + build/clippy/test for the
# transformer, rwkv, stoicheia, clt and diffusion feature sets, plus bare
# and all-software-features builds.
#
# Tiers:
#   ./scripts/preflight.ps1        # FAST (default): full STABLE mirror + MSRV
#                                   # (1.91) fmt + clippy. MSRV clippy compiles on
#                                   # 1.91, so it catches version-specific lints
#                                   # (e.g. clippy::suboptimal_flops, which fired
#                                   # only on 1.91) without paying for MSRV
#                                   # build/test.
#   ./scripts/preflight.ps1 -Ci    # FULL mirror: every ci.yml step on BOTH 1.91
#                                   # and stable. Literal "green preflight =
#                                   # green CI"; use before important pushes or
#                                   # after MSRV-sensitive changes.
#   ./scripts/preflight.ps1 -Full   # also run the bench_hook_* CPU benches
#                                   # (composes with -Ci). See note below.
#
# Lane selection (ad-hoc checks, NOT a push gate):
#   ./scripts/preflight.ps1 -ListLanes        # print the slugs, then exit
#   ./scripts/preflight.ps1 -Only clt         # just the clt lane
#   ./scripts/preflight.ps1 -Only clt,sae     # several
#   ./scripts/preflight.ps1 -Skip training    # everything but one
#
# This exists so a one-off check is a slug rather than a hand-written feature
# string. Assembling `--no-default-features --features ...` by hand is how the
# wrong lane gets run: `--no-default-features` drops `cuda` along with
# everything else (it is in `default`), so a command meant to isolate a
# feature also silently drops to CPU. The script already knows the right
# flags for every lane; -Only just lets you name one.
#
# A filtered run freshens no toolchain, runs stable only unless -Ci, and is
# NOT "green preflight = green CI". Run the script bare before pushing.
#
# The `rustup update stable` at the top is the whole point: CI tracks rolling
# stable, so a dry-run on a stale local toolchain can pass while CI fails on a
# lint only the newer compiler knows (this is how clippy::suboptimal_flops from
# Rust 1.96 once broke a clean main). Freshen first, then lint.
#
# bench_hook_* note: these are timing benchmarks, not correctness checks. They
# SKIP on CI because the gated meta-llama/Llama-3.2-1B isn't cached on the
# runners — so they are not part of "green CI", and the default/-Ci paths skip
# them (`--skip bench_hook`) for parity. Locally the model IS cached, so -Full
# runs them (~slow); do that only when adding a new model family, the change
# that can shift the benchmarked forward/hook paths.
#
# Bypass: not recommended — if you must, just don't run it (convention, not an
#         enforced git hook).

param(
    [switch]$Full,
    [switch]$Ci,
    [string[]]$Only = @(),
    [string[]]$Skip = @(),
    [switch]$ListLanes
)

$ErrorActionPreference = "Stop"

# Run a scriptblock step (used for the toolchain-setup commands that take no
# per-lane parameters), halting the whole preflight on a non-zero exit.
function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Run `cargo +<toolchain> <args...>` as a named step. Arguments are passed as an
# explicit array (not captured in a scriptblock) so toolchain/feature values are
# bound by value — no PowerShell closure surprises.
function Invoke-Cargo {
    param(
        [string]$Name,
        [string]$Tc,
        [string[]]$CargoArgs,
        [string[]]$Lane = @()
    )
    if (-not (Test-LaneSelected $Lane)) {
        Write-Host "--- skipped: $Name" -ForegroundColor DarkGray
        return
    }
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & cargo "+$Tc" @CargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Feature sets and the all-software set. Keep in sync with ci.yml.
#
# `Id` is the -Only/-Skip slug; `Features` is what cargo actually receives.
# The slug is the short name of the lane, not the feature string, so
# `-Only clt` reaches the `clt,transformer` lane without anyone retyping the
# companion feature that lane happens to need.
$lanes = @(
    @{ Id = 'transformer';  Features = 'transformer' }
    @{ Id = 'rwkv';         Features = 'rwkv,rwkv-tokenizer' }
    @{ Id = 'stoicheia';    Features = 'stoicheia' }
    @{ Id = 'clt';          Features = 'clt,transformer' }
    @{ Id = 'quantized';    Features = 'quantized,transformer' }
    @{ Id = 'sae';          Features = 'sae,transformer' }
    @{ Id = 'mmap';         Features = 'mmap,transformer' }
    @{ Id = 'diffusion';    Features = 'diffusion' }
    @{ Id = 'memory';       Features = 'memory' }
    @{ Id = 'memory-debug'; Features = 'memory-debug' }
    @{ Id = 'training';     Features = 'training' }
)
$allSoftware = "transformer,rwkv,rwkv-tokenizer,diffusion,clt,sae,stoicheia,quantized,training"

# Slugs that are not per-feature lanes: the formatting check, the bare
# (no-features) builds, and the all-software builds/doctests.
$metaLanes = @('fmt', 'bare', 'all')
$knownLanes = @($lanes | ForEach-Object { $_.Id }) + $metaLanes

if ($ListLanes) {
    Write-Host "Lanes (use with -Only / -Skip):`n" -ForegroundColor Cyan
    foreach ($l in $lanes) {
        Write-Host ("  {0,-13} --features {1}" -f $l.Id, $l.Features)
    }
    Write-Host ''
    Write-Host ("  {0,-13} cargo fmt --check" -f 'fmt')
    Write-Host ("  {0,-13} bare rustdoc + no-default-features build" -f 'bare')
    Write-Host ("  {0,-13} all-software rustdoc, doctests and build" -f 'all')
    exit 0
}

# Reject an unknown slug rather than silently running nothing. A typo in
# `-Only` would otherwise look like a clean pass over zero lanes, which is the
# worst possible failure mode for a script people trust before pushing.
foreach ($s in ($Only + $Skip)) {
    if ($knownLanes -notcontains $s) {
        Write-Host "Unknown lane '$s'. Known lanes:" -ForegroundColor Red
        Write-Host ('  ' + ($knownLanes -join ', '))
        exit 2
    }
}

$filtered = ($Only.Count -gt 0) -or ($Skip.Count -gt 0)

# Is a step's lane selected? Steps carry one or more tags; a step runs when any
# of its tags is selected. Untagged steps always run.
function Test-LaneSelected {
    param([string[]]$Tags)

    if ($Tags.Count -eq 0) { return $true }
    foreach ($t in $Tags) { if ($Skip -contains $t) { return $false } }
    if ($Only.Count -eq 0) { return $true }
    foreach ($t in $Tags) { if ($Only -contains $t) { return $true } }
    return $false
}

# Skip the bench_hook_* benches unless -Full (see header note).
$benchArgs = if ($Full) { @() } else { @("--", "--skip", "bench_hook") }

# Run CI's lanes for one toolchain. With -LintOnly, run only fmt + the clippy
# lanes (clippy compiles + lints on that toolchain) and skip the build/test
# steps — used for the MSRV toolchain on the fast default path.
function Invoke-Lanes {
    param(
        [string]$Tc,
        [bool]$LintOnly
    )

    Invoke-Cargo "[$Tc] Formatting" $Tc @("fmt", "--check") -Lane 'fmt'

    foreach ($l in $lanes) {
        Invoke-Cargo "[$Tc] Clippy ($($l.Features))" $Tc `
            @("clippy", "--all-targets", "--no-default-features", "--features", $l.Features, "--", "-W", "clippy::pedantic") `
            -Lane $l.Id
    }

    # Rustdoc, per feature set, mirroring ci.yml's own loop. `#![deny(warnings)]`
    # promotes rustdoc::broken_intra_doc_links to an error, but only for the
    # features enabled in that run: a link to a feature-gated item resolves when
    # its feature is on and dangles when it is off, so an --all-features pass
    # structurally cannot see it. The bare run (no features) is the strictest
    # case. Kept in the lint phase, before the -LintOnly return, so the MSRV
    # toolchain runs it too on the fast default path -- rustdoc lints are
    # version-specific in the same way clippy's are, which is the whole reason
    # the MSRV clippy lanes are here. Each run is seconds (`--no-deps`).
    Invoke-Cargo "[$Tc] Rustdoc (bare)" $Tc @("doc", "--no-default-features", "--no-deps") -Lane 'bare'
    foreach ($l in $lanes) {
        Invoke-Cargo "[$Tc] Rustdoc ($($l.Features))" $Tc `
            @("doc", "--no-default-features", "--features", $l.Features, "--no-deps") `
            -Lane $l.Id
    }
    Invoke-Cargo "[$Tc] Rustdoc (all software features)" $Tc `
        @("doc", "--no-default-features", "--features", "$allSoftware,memory", "--no-deps") -Lane 'all'

    if ($LintOnly) { return }

    # Builds + tests, mirroring ci.yml step-for-step.
    Invoke-Cargo "[$Tc] Build (transformer)" $Tc `
        @("build", "--no-default-features", "--features", "transformer") -Lane 'transformer'
    Invoke-Cargo "[$Tc] Tests (transformer)" $Tc `
        (@("test", "--no-default-features", "--features", "transformer") + $benchArgs) -Lane 'transformer'

    Invoke-Cargo "[$Tc] Build (RWKV)" $Tc `
        @("build", "--no-default-features", "--features", "rwkv,rwkv-tokenizer") -Lane 'rwkv'
    Invoke-Cargo "[$Tc] Tests (RWKV)" $Tc `
        @("test", "--no-default-features", "--features", "rwkv,rwkv-tokenizer") -Lane 'rwkv'

    Invoke-Cargo "[$Tc] Build (Stoicheia)" $Tc `
        @("build", "--no-default-features", "--features", "stoicheia") -Lane 'stoicheia'
    Invoke-Cargo "[$Tc] Tests (Stoicheia)" $Tc `
        @("test", "--no-default-features", "--features", "stoicheia", "--lib", "--test", "stoicheia_analysis", "--test", "validate_stoicheia") -Lane 'stoicheia'

    Invoke-Cargo "[$Tc] Build (CLT)" $Tc `
        @("build", "--no-default-features", "--features", "clt,transformer") -Lane 'clt'
    Invoke-Cargo "[$Tc] Tests (CLT)" $Tc `
        @("test", "--no-default-features", "--features", "clt,transformer", "--lib") -Lane 'clt'

    # Quantized loading (bnb/AWQ/GPTQ dequant); parity tests are #[ignore], so
    # this compiles the test binary to catch signature/API breaks.
    Invoke-Cargo "[$Tc] Build (Quantized)" $Tc `
        @("build", "--no-default-features", "--features", "quantized,transformer") -Lane 'quantized'
    Invoke-Cargo "[$Tc] Tests (Quantized)" $Tc `
        @("test", "--no-default-features", "--features", "quantized,transformer", "--lib", "--test", "validate_quantized_loading") -Lane 'quantized'

    # validate_plt_gemma needs clt+sae+transformer together; no single-feature
    # lane builds that test binary. The test is #[ignore], so compile-check it.
    Invoke-Cargo "[$Tc] Tests compile (clt+sae+transformer)" $Tc `
        @("test", "--no-default-features", "--features", "clt,sae,transformer", "--test", "validate_plt_gemma", "--no-run") -Lane @('clt','sae')

    Invoke-Cargo "[$Tc] Build (Diffusion + examples)" $Tc `
        @("build", "--no-default-features", "--features", "diffusion", "--examples") -Lane 'diffusion'
    Invoke-Cargo "[$Tc] Tests (Diffusion)" $Tc `
        @("test", "--no-default-features", "--features", "diffusion", "--lib", "--test", "validate_mdlm_forward", "--test", "validate_othello_forward", "--test", "validate_patch_at") -Lane 'diffusion'

    # validate_bidirectional_sampler needs transformer + diffusion together; no
    # single-feature lane builds it, so compile-check it explicitly.
    Invoke-Cargo "[$Tc] Tests compile (transformer+diffusion)" $Tc `
        @("test", "--no-default-features", "--features", "transformer,diffusion", "--test", "validate_bidirectional_sampler", "--test", "validate_bidirectional_forward", "--no-run") -Lane @('transformer','diffusion')

    # Memory measurement (delegated to hypomnesis); live CUDA checks are
    # #[ignore], the CPU invariant runs here.
    Invoke-Cargo "[$Tc] Build (memory)" $Tc `
        @("build", "--no-default-features", "--features", "memory") -Lane 'memory'
    Invoke-Cargo "[$Tc] Tests (memory)" $Tc `
        @("test", "--no-default-features", "--features", "memory", "--lib", "--test", "validate_memory") -Lane 'memory'

    # Checkpointable AdamW (`training`, default-off). Adding the feature to
    # `$featureSets` only buys a clippy lane; the parity tests need running, and
    # they are the claim the module rests on (our transcribed update rule vs
    # candle's own trajectory). Without this step a parity failure would pass
    # preflight and fail CI -- precisely the mirror gap preflight exists to close.
    Invoke-Cargo "[$Tc] Build (training)" $Tc `
        @("build", "--no-default-features", "--features", "training") -Lane 'training'
    Invoke-Cargo "[$Tc] Tests (training)" $Tc `
        @("test", "--no-default-features", "--features", "training", "--lib", "--test", "validate_optim_parity") -Lane 'training'

    # Doctest every feature-gated public example in one pass (clt/sae/rwkv were
    # uncovered by the old transformer+memory-only lane). Mirrors ci.yml §1.10.
    Invoke-Cargo "[$Tc] Doctests (all software features)" $Tc `
        @("test", "--no-default-features", "--features", ($allSoftware + ",memory"), "--doc") -Lane 'all'

    Invoke-Cargo "[$Tc] Build (no default features)" $Tc `
        @("build", "--no-default-features") -Lane 'bare'
    Invoke-Cargo "[$Tc] Build (all software features)" $Tc `
        @("build", "--no-default-features", "--features", $allSoftware) -Lane 'all'
}

# --- Freshen toolchains (match CI's rolling stable; ensure MSRV present) ---
#
# A filtered run skips all of this. Freshening is minutes of network for a
# guarantee (`local stable == CI stable`) that a partial run cannot make
# anyway, and an ad-hoc lane check should be seconds.
if (-not $filtered) {
    Invoke-Step "Update stable toolchain" { rustup update stable }
    Invoke-Step "Ensure MSRV 1.91 toolchain" { rustup toolchain install 1.91 }
    Invoke-Step "Ensure 1.91 components (clippy, rustfmt)" { rustup component add clippy rustfmt --toolchain 1.91 }
    Write-Host "Using:" -ForegroundColor Yellow
    & rustc "+stable" --version
    & rustc "+1.91" --version

    Invoke-Step "Pick up latest hf-fetch-model patch" { cargo update -p hf-fetch-model }
}
else {
    Write-Host "Filtered run: skipping toolchain freshening and the hf-fetch-model bump." -ForegroundColor Yellow
    if ($Only.Count -gt 0) { Write-Host ("  -Only " + ($Only -join ',')) -ForegroundColor Yellow }
    if ($Skip.Count -gt 0) { Write-Host ("  -Skip " + ($Skip -join ',')) -ForegroundColor Yellow }
}

# Stable: always the full lane set. MSRV 1.91: fmt + clippy by default; the full
# lane set only under -Ci. A filtered run stays on stable unless -Ci asks
# otherwise, since the MSRV pass doubles the wait for an ad-hoc check.
Invoke-Lanes -Tc "stable" -LintOnly $false
if ((-not $filtered) -or $Ci) {
    Invoke-Lanes -Tc "1.91" -LintOnly (-not $Ci)
}

if ($filtered) {
    Write-Host "`nSelected lanes passed. This was a PARTIAL run and is not a push gate." -ForegroundColor Yellow
    Write-Host "Run './scripts/preflight.ps1' with no filter before pushing." -ForegroundColor Yellow
}
elseif ($Ci) {
    Write-Host "`nAll preflight checks passed (full 1.91 + stable mirror) — safe to push." -ForegroundColor Green
}
else {
    Write-Host "`nAll preflight checks passed (full stable mirror + MSRV clippy)." -ForegroundColor Green
    Write-Host "Run './scripts/preflight.ps1 -Ci' for the full both-toolchain mirror before important pushes." -ForegroundColor Yellow
}

# Nudge: how stale is the #[ignore]d oracle/parity suite? (reads RESURRECTION.md;
# runs nothing, downloads nothing). Refresh with scripts/resurrect.ps1.
# Suppressed on a filtered run: the nudge is about push readiness, which a
# partial run is not claiming, and its table would bury the lane output.
if (-not $filtered) {
    Write-Host ''
    & (Join-Path $PSScriptRoot 'resurrect.ps1') -Status
}
