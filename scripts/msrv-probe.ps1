# MSRV probe: build candle-mi the way a NEW USER gets it, on the MSRV toolchain.
#
# Every other lane builds candle-mi's own Cargo.lock (`--locked`), which proves
# that THIS lockfile builds on the MSRV. It says nothing about what a consumer
# gets from `cargo add candle-mi`: a fresh resolve that can pick newer
# dependency releases than the lockfile holds. Cargo's MSRV-aware resolver
# steers around releases whose declared `rust-version` is too new, but not
# around a dependency that declares none and still needs a newer compiler.
# Two such floors were invisible to every lane:
#   - candle-core 0.11.1 (no `rust-version`) needs Rust 1.94 on x86_64;
#   - hf-xet >= 1.7 (no `rust-version`) pulls sysinfo 0.39, which needs 1.95.
#
# This script creates a throwaway consumer crate outside the repository, with
# `rust-version` set to candle-mi's (read from Cargo.toml), depending on
# candle-mi by path with every feature except those lane-parity excludes
# (`default`, `cuda`, `metal`: they need hardware or a GPU toolkit). It then
# resolves WITHOUT candle-mi's lockfile and builds (`cargo check`) on the MSRV
# toolchain. It fails if the resolver had to pick a release that declares a
# newer Rust, or if the build fails.
#
# Run by preflight.ps1 (unfiltered runs) and by ci.yml's `msrv-probe` job,
# which also runs in the weekly scheduled canary, so an upstream release that
# silently raises the floor is caught between pushes. Needs network (it
# resolves against crates.io). ~1-2 min warm, longer cold.
#
#   ./scripts/msrv-probe.ps1                 # MSRV from Cargo.toml
#   ./scripts/msrv-probe.ps1 -Toolchain 1.96 # probe another toolchain

param(
    [string]$Toolchain = ''
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$cargoToml = Get-Content -Raw -LiteralPath (Join-Path $root 'Cargo.toml')

$msrv = [regex]::Match($cargoToml, '(?m)^rust-version\s*=\s*"([^"]+)"').Groups[1].Value
if (-not $msrv) { throw "msrv-probe: no rust-version found in Cargo.toml" }
if (-not $Toolchain) { $Toolchain = $msrv }

# The [features] keys, minus lane-parity's `excluded` list. The two helpers are
# lane-parity.ps1's, verbatim, so both scripts read Cargo.toml the same way.
function Get-TomlSection {
    param([string]$Text, [string]$Header)

    $m = [regex]::Match($Text, '(?ms)^\[' + [regex]::Escape($Header) + '\]\s*$(.*?)(?=^\[|\z)')
    if (-not $m.Success) { throw "Cargo.toml has no [$Header] table" }
    return $m.Groups[1].Value
}
function Get-TomlStringArray {
    param([string]$Section, [string]$Key, [string]$Where)

    $m = [regex]::Match($Section, '(?ms)^' + [regex]::Escape($Key) + '\s*=\s*\[(.*?)\]')
    if (-not $m.Success) { throw "Cargo.toml [$Where] has no '$Key' array" }
    return @([regex]::Matches($m.Groups[1].Value, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
}
$features = @([regex]::Matches((Get-TomlSection $cargoToml 'features'), '(?m)^([A-Za-z0-9_-]+)\s*=') |
    ForEach-Object { $_.Groups[1].Value })
$excluded = Get-TomlStringArray (Get-TomlSection $cargoToml 'package.metadata.lane-parity') 'excluded' 'package.metadata.lane-parity'
$probeFeatures = @($features | Where-Object { $excluded -cnotcontains $_ })
if ($probeFeatures.Count -eq 0) { throw "msrv-probe: no features left to probe" }

# A fresh directory outside the repository, so no workspace, lockfile or
# `.cargo/config` of candle-mi's applies to the consumer.
$probe = Join-Path ([IO.Path]::GetTempPath()) ("candle-mi-msrv-probe-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path (Join-Path $probe 'src') | Out-Null

try {
    # TOML literal string ('...') so Windows backslashes need no escaping.
    $featureList = ($probeFeatures | ForEach-Object { '"' + $_ + '"' }) -join ', '
    @"
[package]
name = "candle-mi-msrv-probe"
version = "0.0.0"
edition = "2024"
rust-version = "$msrv"
publish = false

[dependencies]
candle-mi = { path = '$root', default-features = false, features = [$featureList] }
"@ | Set-Content -LiteralPath (Join-Path $probe 'Cargo.toml') -Encoding utf8
    'fn main() {}' | Set-Content -LiteralPath (Join-Path $probe 'src/main.rs') -Encoding utf8

    Write-Host "MSRV probe: candle-mi (rust-version $msrv) as a fresh dependency, toolchain $Toolchain" -ForegroundColor Cyan
    Write-Host "  features: $($probeFeatures -join ',')"
    Write-Host "  in: $probe"

    Push-Location $probe
    try {
        # Resolve. Cargo prints "(requires Rust X)" next to any package it had
        # to pick despite a declared rust-version above the probe's.
        # Cargo writes its progress to stderr. Windows PowerShell 5.1 turns each
        # merged stderr line into a terminating error under "Stop", so relax the
        # preference for this one call; the exit code is checked explicitly.
        $saved = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $resolve = & cargo "+$Toolchain" generate-lockfile 2>&1 | ForEach-Object { "$_" }
            $resolveExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $saved
        }
        $resolve | ForEach-Object { Write-Host "  $_" }
        if ($resolveExit -ne 0) { throw "msrv-probe: resolution failed (exit $resolveExit)" }
        $tooNew = @($resolve | Where-Object { $_ -match 'requires Rust' })
        if ($tooNew.Count -gt 0) {
            Write-Host "MSRV probe FAILED: a fresh resolve needs a newer Rust than $msrv for:" -ForegroundColor Red
            $tooNew | ForEach-Object { Write-Host "  $($_.Trim())" }
            exit 1
        }

        # Build. This is what catches a dependency that declares no
        # rust-version at all and still needs a newer compiler.
        & cargo "+$Toolchain" check
        if ($LASTEXITCODE -ne 0) {
            Write-Host "MSRV probe FAILED: a fresh resolve of candle-mi does not build on $Toolchain." -ForegroundColor Red
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $probe -ErrorAction SilentlyContinue
}

Write-Host "MSRV probe OK: a fresh resolve of candle-mi builds on $Toolchain." -ForegroundColor Green
exit 0
