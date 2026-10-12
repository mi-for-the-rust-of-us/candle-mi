# Lane parity: fail when the hand-maintained lane lists disagree (candle-mi#2).
#
# The feature lanes are written out in four places: .github/workflows/ci.yml,
# .github/workflows/publish.yml, scripts/preflight.ps1, and Cargo.toml's
# [package.metadata.docs.rs] list. Nothing compared them, and they drifted
# twice: the docs.rs list omitted `training` until v0.1.23, so `optim::AdamW`
# shipped in 0.1.22 with no docs.rs pages, and the `fused-attn` lane reached
# ci.yml and preflight but not publish.yml. This script turns that drift into a
# failure instead of a finding.
#
# Checks:
#   1. publish.yml runs exactly ci.yml's cargo commands, plus its own
#      `cargo package` and `cargo publish`.
#   2. preflight.ps1 runs exactly ci.yml's cargo commands. They are read from
#      its -PrintCommands switch, so nothing here parses PowerShell.
#   3. Every feature in Cargo.toml's [features] is in the docs.rs list and in at
#      least one ci.yml lane, except those listed in
#      [package.metadata.lane-parity] `excluded`; the docs.rs list names nothing
#      else.
#
# Commands are compared after normalisation: `--locked` (CI only) is dropped,
# each `--features` list is sorted, and preflight's trailing
# `-- --skip bench_hook` (see the bench_hook_* note in its header) is dropped.
#
# Run by preflight.ps1 (unfiltered runs) and by ci.yml's `lanes` job. Seconds;
# it runs no cargo command.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

# Normalise one cargo command line into the string the comparisons use.
function ConvertTo-CanonicalCommand {
    param([string]$Line)

    $tokens = @($Line.Trim() -split '\s+' |
        ForEach-Object { $_.Trim('"', "'") } |
        Where-Object { $_ -ne '' -and $_ -ne '--locked' })

    # Preflight-only bench filter, appended to the transformer tests.
    $n = $tokens.Count
    if ($n -ge 4 -and $tokens[$n - 3] -eq '--' -and $tokens[$n - 2] -eq '--skip' -and $tokens[$n - 1] -eq 'bench_hook') {
        $tokens = $tokens[0..($n - 4)]
    }

    for ($i = 0; $i -lt $tokens.Count - 1; $i++) {
        if ($tokens[$i] -eq '--features') {
            $tokens[$i + 1] = (($tokens[$i + 1] -split ',') | Sort-Object) -join ','
        }
    }
    return ($tokens -join ' ')
}

# The cargo commands a workflow runs: every single-line `run: cargo ...`, plus
# the per-feature rustdoc loop expanded into one `cargo doc` per entry. A loop
# that no longer has the expected shape is an error rather than zero commands,
# so restructuring it cannot silently switch this check off.
function Get-WorkflowCommands {
    param([string]$Path)

    $text = Get-Content -Raw -LiteralPath $Path
    $commands = [System.Collections.Generic.List[string]]::new()
    foreach ($m in [regex]::Matches($text, '(?m)^\s+run:\s+(cargo\s.+?)\s*$')) {
        $commands.Add((ConvertTo-CanonicalCommand $m.Groups[1].Value))
    }

    $loop = [regex]::Match($text, '(?s)for f in (.+?); do\s+.*?cargo doc --locked --no-default-features \$\{f:\+--features "\$f"\} --no-deps')
    if (-not $loop.Success) {
        throw "$Path has no per-feature rustdoc loop of the expected shape; if it was restructured, update scripts/lane-parity.ps1"
    }
    foreach ($m in [regex]::Matches($loop.Groups[1].Value, '"([^"]*)"')) {
        $f = $m.Groups[1].Value
        $line = if ($f -eq '') {
            'cargo doc --no-default-features --no-deps'
        }
        else {
            "cargo doc --no-default-features --features $f --no-deps"
        }
        $commands.Add((ConvertTo-CanonicalCommand $line))
    }
    return $commands
}

# The body of a `[header]` table in a TOML file, up to the next table.
function Get-TomlSection {
    param([string]$Text, [string]$Header)

    $m = [regex]::Match($Text, '(?ms)^\[' + [regex]::Escape($Header) + '\]\s*$(.*?)(?=^\[|\z)')
    if (-not $m.Success) { throw "Cargo.toml has no [$Header] table" }
    return $m.Groups[1].Value
}

# The strings of a `key = [ "a", "b" ]` array inside one TOML table body.
function Get-TomlStringArray {
    param([string]$Section, [string]$Key, [string]$Where)

    $m = [regex]::Match($Section, '(?ms)^' + [regex]::Escape($Key) + '\s*=\s*\[(.*?)\]')
    if (-not $m.Success) { throw "Cargo.toml [$Where] has no '$Key' array" }
    return @([regex]::Matches($m.Groups[1].Value, '"([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
}

$failures = [System.Collections.Generic.List[string]]::new()

# Record every command one side has and the other lacks (case-sensitive).
function Compare-CommandSets {
    param([string]$LeftName, [string[]]$Left, [string]$RightName, [string[]]$Right)

    foreach ($c in ($Left | Sort-Object -Unique)) {
        if ($Right -cnotcontains $c) { $failures.Add("in $LeftName, not in ${RightName}: $c") }
    }
    foreach ($c in ($Right | Sort-Object -Unique)) {
        if ($Left -cnotcontains $c) { $failures.Add("in $RightName, not in ${LeftName}: $c") }
    }
}

try {
    $ci = @(Get-WorkflowCommands (Join-Path $root '.github/workflows/ci.yml'))
    $publishAll = @(Get-WorkflowCommands (Join-Path $root '.github/workflows/publish.yml'))
    $publish = @($publishAll | Where-Object { $_ -notmatch '^cargo (package|publish) ' })
    $preflight = @(& (Join-Path $PSScriptRoot 'preflight.ps1') -PrintCommands |
        ForEach-Object { ConvertTo-CanonicalCommand $_ })

    # Checks 1 and 2.
    Compare-CommandSets 'ci.yml' $ci 'publish.yml' $publish
    Compare-CommandSets 'ci.yml' $ci 'preflight.ps1' $preflight

    # Check 3.
    $cargoToml = Get-Content -Raw -LiteralPath (Join-Path $root 'Cargo.toml')
    $features = @([regex]::Matches((Get-TomlSection $cargoToml 'features'), '(?m)^([A-Za-z0-9_-]+)\s*=') |
        ForEach-Object { $_.Groups[1].Value })
    $docsRs = Get-TomlStringArray (Get-TomlSection $cargoToml 'package.metadata.docs.rs') 'features' 'package.metadata.docs.rs'
    $excluded = Get-TomlStringArray (Get-TomlSection $cargoToml 'package.metadata.lane-parity') 'excluded' 'package.metadata.lane-parity'

    $ciFeatures = @($ci | ForEach-Object {
            $m = [regex]::Match($_, '--features (\S+)')
            if ($m.Success) { $m.Groups[1].Value -split ',' }
        } | Sort-Object -Unique)

    foreach ($e in $excluded) {
        if ($features -cnotcontains $e) { $failures.Add("lane-parity 'excluded' names '$e', which is not a feature") }
    }
    foreach ($f in $features) {
        if ($excluded -ccontains $f) { continue }
        if ($docsRs -cnotcontains $f) { $failures.Add("feature '$f' is in neither the docs.rs list nor lane-parity 'excluded'") }
        if ($ciFeatures -cnotcontains $f) { $failures.Add("feature '$f' has no ci.yml lane and is not in lane-parity 'excluded'") }
    }
    foreach ($d in $docsRs) {
        if ($features -cnotcontains $d) { $failures.Add("docs.rs list names '$d', which is not a feature") }
        elseif ($excluded -ccontains $d) { $failures.Add("docs.rs list names '$d', which lane-parity 'excluded' also names") }
    }
}
catch {
    Write-Host "lane parity: $_" -ForegroundColor Red
    exit 1
}

if ($failures.Count -gt 0) {
    Write-Host "Lane parity FAILED ($($failures.Count)):" -ForegroundColor Red
    foreach ($f in $failures) { Write-Host "  $f" }
    exit 1
}

Write-Host ("Lane parity OK: ci.yml, publish.yml and preflight.ps1 run the same {0} cargo commands; " -f ($ci | Sort-Object -Unique).Count) -NoNewline
Write-Host ("docs.rs and CI cover every feature but {0}." -f ($excluded -join ', ')) -ForegroundColor Green
exit 0
