# resurrect.ps1 — run the #[ignore]d oracle/parity suite locally and stamp
# RESURRECTION.md with a PER-TEST "last verified" date and wall-clock.
#
# The oracle/parity tests are #[ignore]d because they need cached HuggingFace
# models (some gated) and, for many, a CUDA GPU with >= 16 GiB VRAM — so they
# never run in GitHub CI (CPU-only runners, no cached models). This runner
# exercises them on a workstation where the models are cached, giving the
# #[ignore]d suite the local safety net the 2026-07-01 audit (§1.1) flagged as
# missing, and writes a committed staleness log.
#
# Staleness is tracked PER ENTRY, not per run: each test carries its own "last
# verified" date (advanced only on a real PASS). So a -Quick run refreshes only
# the three entries it ran; everything else keeps its true (older) date. This keeps
# the report honest and decouples staleness from the tier you happened to run.
#
# Tiers (how much to run now — nothing more):
#   -Quick    cheap ungated smoke: CPU encoder-parity (clt_qwen3, plt_gemma)
#             plus the GPU patch_at guard, which downloads nothing — ~6 min
#   (default) everything EXCEPT the two slow outliers (Mistral-7B CPU forward and
#             the anacrousis 28x15 matrix) — ~40-50 min
#   -Full     literally everything, incl. the two slow outliers — ~1.5-3 h
#
# Selecting entries (so refreshing one costs a minute, not a tier):
#   -Only <tokens>  run ONLY these. 1-based numbers as printed by -List, or the
#                   stable per-entry slugs: `-Only longrope`, `-Only 12`,
#                   `-Only clt,sae`. Prefer slugs in anything you write down —
#                   inserting an entry renumbers everything after it. Matching is
#                   exact, never prefix (`clt` and `clt_qwen3` both exist), and an
#                   unknown token is an error, never a silent empty run.
#   -Skip <tokens>  drop these from whatever the tier or -Only chose.
#                   `-Skip longrope` is the usual one: that entry alone is ~15 of
#                   the ~44 min default tier, so skipping it gives ~28 min.
#   Any selector makes the run PARTIAL, and it stamps "partial (N of 21: ...)"
#   rather than a tier name — a three-entry run must never later read as full
#   coverage.
#
# Read-only:
#   -Status   print how stale the suite is (oldest entry, entries past the
#             threshold, any failing) and exit — runs NOTHING, downloads NOTHING.
#             `-StaleDays N` sets the threshold (default 50).
#   -List     print the number/slug map with each entry's tier and last-verified
#             state, then exit. Start here to find the token you want.
#
# Timing: each entry's end-to-end wall-clock is recorded in RESURRECTION.md (a
#   per-entry column, advanced only on a PASS — same rule as "last verified") and
#   printed as a "slowest first" summary. `-SpillWarnSeconds N` (default 300) flags
#   a step slow enough to suspect VRAM spill to shared memory (e.g. longrope/
#   Phi-3.5-mini at F32 overflows a 16 GiB card and crawls at ~15x its warm time).
#
# Spill measurement: entries marked `Spill = $true` (and everything under
#   -SpillProbe) run wrapped in `hmn spill --json` (hypomnesis), recording the
#   growth of resident shared-system memory above its benign baseline, how long
#   the spill lasted, and peak demand. That growth is the signal that matters:
#   during a spill NVML "used" pins near capacity and cannot show how far over
#   budget a run went. `-SpillIntervalMs N` sets the sampling rate (default 100,
#   matching hypomnesis; below ~50 ms buys no resolution). Absent `hmn`, the run
#   proceeds with a note — instrumentation, never a gate.
#
# Device policy: GPU tests run on the GPU; CPU-parity tests (*_cpu, plt,
# clt_qwen3) run on CPU by design. Runs with the default feature set so `cuda`
# stays ON; each entry adds its extra feature(s).

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Status,
    [int]$StaleDays = 50,
    [int]$SpillWarnSeconds = 300,
    # Force WDDM spill sampling on every entry, not just those marked `Spill`.
    # Requires `hmn` (hypomnesis's CLI) on PATH; skipped with a note if absent.
    [switch]$SpillProbe,
    # Spill-sampling interval passed through to `hmn spill --interval`. 100 ms
    # matches hypomnesis's own default. Going below ~50 ms buys no resolution
    # (the GPU counters only update on driver cadence) and costs PDH queries;
    # raising it cuts that cost but risks missing a brief peak. The 1 ms floor
    # mirrors hmn's own `range(1..)`; the ceiling is a local sanity guard.
    [ValidateRange(1, 60000)]
    [int]$SpillIntervalMs = 100,
    # Run only these entries: 1-based numbers as shown by -List, or stable slugs
    # (`longrope`, `clt`, `sae`). Numbers are the convenient form at the prompt;
    # slugs are the durable one, because inserting an entry renumbers everything
    # after it. Prefer slugs in commit messages and runbooks. Mixing is fine:
    # `-Only 12,sae`. Overrides the tier switches.
    #
    # Typed `[string[]]`, which is load-bearing rather than cosmetic: at the
    # prompt PowerShell parses an unquoted `a,b` as an ARRAY, so the comma form
    # this comment documents failed to bind while `-Only` was a bare [string]
    # ("Impossible de convertir la valeur en type System.String"). Accepting
    # both shapes means `-Only clt,sae` and `-Only "clt,sae"` behave the same,
    # and it matches `preflight.ps1 -Only`, which takes `[string[]]` too.
    [string[]]$Only,
    # Drop these entries from whatever the tier or -Only selected. Same syntax.
    # `-Skip longrope` is the common case: it removes the single 14-minute
    # VRAM-spill step and turns a ~44 min Default run into ~28 min.
    [string[]]$Skip,
    # Print the number/slug map with each entry's tier and last-verified state,
    # then exit. Runs nothing, downloads nothing.
    [switch]$List
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$mdPath = Join-Path $repoRoot 'RESURRECTION.md'

# Canonical entry list. `Features` are ADDED to the default set (cuda,transformer).
# Quick=$true -> included in -Quick; FullOnly=$true -> only in -Full;
# SlowSkip -> a `--skip <pat>` applied in the default tier (dropped in -Full).
# Lib=$true + Filter -> run `#[ignore]`d unit tests in the library instead of an
#   integration test binary (`--lib -- <Filter>`), for guards that need no model
#   and so have no integration-test home. `Tests` stays empty for those.
$entries = @(
    @{ Id = 'clt_qwen3'; Name = 'clt_qwen3 (encoder parity)'; Features = 'clt';                         Tests = @('validate_clt_qwen3');                             Device = 'CPU';     Models = 'bluelightai/clt-qwen3-1.7b-base-20k (~240 MiB)';        Quick = $true }
    @{ Id = 'plt_gemma'; Name = 'plt_gemma (encoder parity)'; Features = 'clt,sae';                     Tests = @('validate_plt_gemma');                             Device = 'CPU';     Models = 'google/gemma-scope-2b-pt-transcoders (~864 MiB, gated)'; Quick = $true }
    @{ Id = 'plt_llama'; Name = 'plt_llama (encoder parity)'; Features = 'clt';                         Tests = @('validate_plt');                                   Device = 'CPU';     Models = 'mntss/transcoder-Llama-3.2-1B (~16 GiB)' }
    @{ Id = 'llama32'; Name = 'llama32 forward';            Features = 'mmap';                        Tests = @('validate_llama32_forward');                       Device = 'CPU+GPU'; Models = 'meta-llama/Llama-3.2-1B (gated)' }
    @{ Id = 'gemma2'; Name = 'gemma2 forward';             Features = 'mmap';                        Tests = @('validate_gemma2_forward');                        Device = 'CPU+GPU'; Models = 'google/gemma-2-2b (gated)' }
    @{ Id = 'phi3'; Name = 'phi3-mini forward';          Features = 'mmap';                        Tests = @('validate_phi3_mini_forward');                     Device = 'CPU+GPU'; Models = 'microsoft/Phi-3-mini-4k-instruct' }
    @{ Id = 'mistral7b'; Name = 'mistral-7b forward';         Features = 'mmap';                        Tests = @('validate_mistral_7b_forward');                    Device = 'CPU+GPU'; Models = 'mistralai/Mistral-7B-v0.1 (gated)'; SlowSkip = '_cpu' }
    @{ Id = 'qwen3'; Name = 'qwen3 forward';              Features = 'mmap';                        Tests = @('validate_qwen3_forward');                         Device = 'CPU+GPU'; Models = 'Qwen/Qwen3-1.7B-Base' }
    @{ Id = 'qwen25coder'; Name = 'qwen2.5-coder forward';      Features = 'mmap';                        Tests = @('validate_qwen25_coder_forward');                  Device = 'CPU+GPU'; Models = 'Qwen/Qwen2.5-Coder-3B-Instruct' }
    @{ Id = 'starcoder2'; Name = 'starcoder2 forward';         Features = 'mmap';                        Tests = @('validate_starcoder2_forward');                    Device = 'CPU+GPU'; Models = 'bigcode/starcoder2-3b (gated)' }
    @{ Id = 'deepseek'; Name = 'deepseek forward';           Features = 'mmap';                        Tests = @('validate_deepseek_forward');                      Device = 'CPU+GPU'; Models = 'deepseek-ai/deepseek-coder-1.3b-base' }
    # Spill = $true: Phi-3.5-mini is 3.82 B params, so F32 weights alone want
    # 14,572 MiB against a 16,311 MiB card (16,052 usable) — under ~1.5 GiB of
    # headroom for activations, and in practice WDDM pages the shortfall over
    # PCIe, which is why this step runs ~15x its neighbours. Sample it so the
    # log records the measured spill instead of inferring it from wall-clock.
    @{ Id = 'longrope'; Name = 'longrope forward';           Features = 'mmap';                        Tests = @('validate_longrope');                              Device = 'GPU';     Models = 'microsoft/Phi-3.5-mini-instruct (~15 GiB)'; Spill = $true }
    @{ Id = 'bidirectional'; Name = 'bidirectional (a2d-qwen2)';  Features = 'diffusion,mmap';              Tests = @('validate_bidirectional_forward');                 Device = 'CPU+GPU'; Models = 'dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm (~1.2 GiB)' }
    @{ Id = 'mdlm'; Name = 'mdlm + othello';             Features = 'diffusion,mmap';              Tests = @('validate_mdlm_forward', 'validate_othello_forward'); Device = 'CPU+GPU'; Models = 'TheQweaker/mdlm-owt-noflash; Othello fixtures (OTHELLO_MDLM_FIXTURES)' }
    @{ Id = 'quantized'; Name = 'quantized (bnb/AWQ/GPTQ)';   Features = 'quantized,mmap';              Tests = @('validate_quantized_loading');                     Device = 'GPU';     Models = 'medmekk/...-bnb-nf4; casperhansen/...-awq; shuyuej/...-GPTQ' }
    @{ Id = 'clt'; Name = 'clt (encode/inject/sweep)';  Features = 'clt,mmap';                    Tests = @('validate_clt');                                   Device = 'GPU';     Models = 'gemma-2-2b + llama-3.2-1b + mntss CLTs (>=16 GiB VRAM)' }
    @{ Id = 'sae'; Name = 'sae (encode/inject/parity)'; Features = 'sae,mmap';                    Tests = @('validate_sae');                                   Device = 'GPU';     Models = 'gemma-2-2b + gemma-scope-2b-pt-res' }
    @{ Id = 'memory'; Name = 'memory (VRAM probe)';        Features = 'memory';                      Tests = @('validate_memory');                                Device = 'GPU';     Models = '(none - allocates a GPU tensor)' }
    @{ Id = 'rwkv'; Name = 'rwkv6 + rwkv7';              Features = 'rwkv,rwkv-tokenizer,mmap';    Tests = @('validate_rwkv6', 'validate_rwkv7');               Device = 'CPU+GPU'; Models = 'RWKV v6-Finch-1B6; RWKV7-Goose-1.5B' }
    @{ Id = 'anacrousis'; Name = 'anacrousis (28x15 matrix)';  Features = 'mmap';                        Tests = @('validate_anacrousis');                            Device = 'GPU';     Models = 'meta-llama/Llama-3.2-1B (gated)'; FullOnly = $true }
    # Appended last on purpose: inserting anywhere above renumbers every entry
    # after it, and the numbers are what `-Only <n>` takes.
    @{ Id = 'patchat'; Name = 'patch_at (CUDA offset-view guard)'; Features = 'diffusion';             Tests = @('validate_patch_at');                              Device = 'GPU';     Models = '(none - OthelloGpt::init from a seed)'; Quick = $true; Lib = $true; Filter = 'cuda_patch' }
)

# --- Read prior per-entry state back out of RESURRECTION.md (Name -> {Date, Wall, Outcome}) ---
function Read-PriorState {
    param([string]$Path)
    $state = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $state }
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -notmatch '^\|') { continue }
        $cells = ($line.Trim().Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
        if ($cells.Count -lt 5) { continue }
        $name = $cells[0]
        if ($name -eq 'Test' -or $name -match '^-+$') { continue }
        # Three generations of row, newest first. 7-col adds Peak spill
        # (Date=3, Wall=4, Spill=5, Outcome=6); 6-col added Wall-clock
        # (Outcome=5); legacy 5-col rows have neither (Outcome=4).
        if ($cells.Count -ge 7) {
            $state[$name] = @{ Date = $cells[3]; Wall = $cells[4]; Spill = $cells[5]; Outcome = $cells[6] }
        }
        elseif ($cells.Count -eq 6) {
            $state[$name] = @{ Date = $cells[3]; Wall = $cells[4]; Spill = '—'; Outcome = $cells[5] }
        }
        else {
            $state[$name] = @{ Date = $cells[3]; Wall = '—'; Spill = '—'; Outcome = $cells[4] }
        }
    }
    return $state
}

function Format-Age {
    param([int]$Days)
    if ($Days -ge 60) { return "~$([math]::Round($Days / 30)) months" }
    return "$Days days"
}

# Human-friendly wall-clock. Forces InvariantCulture so the decimal point stays a
# '.' on a fr-FR host (a ',' would read oddly next to the file's other numbers).
function Format-Duration {
    param([double]$Seconds)
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    if ($Seconds -lt 90) { return ([math]::Round($Seconds, 1)).ToString($inv) + 's' }
    $mins = [int][math]::Floor($Seconds / 60)
    $secs = [int]($Seconds - $mins * 60)
    if ($mins -lt 60) { return ('{0}m{1:D2}s' -f $mins, $secs) }
    $hrs = [int][math]::Floor($mins / 60)
    $rem = $mins - $hrs * 60
    return ('{0}h{1:D2}m' -f $hrs, $rem)
}

# --- Render an `hmn spill --json` SpillReport as one table cell ---
#
# Three figures, because two of them alone mislead:
#   <spill> = peak_shared MINUS baseline_shared, i.e. growth above the benign
#             staging/upload-heap baseline every process carries. The raw peak
#             would overstate every spill by that baseline.
#   <dur>   = how long the spill lasted; a spill over 90% of a run is a
#             different animal from a brief transient.
#   <peak>  = peak_dedicated + that growth, i.e. roughly what the step actually
#             WANTED. Without it the log says how much overflowed but not how
#             much was asked for, which is the number that decides whether a
#             narrower dtype would fit. It is a sum of two peaks that need not
#             be simultaneous, so treat it as an upper bound (legend says so).
#
# Distinguishes three outcomes that must not be conflated: not measured ('—'),
# measured and clean ('none'), measured and spilled. Per hmn's own guidance,
# `measurable` is checked before trusting `spilled: false`.
function Format-Spill {
    param($Report)
    if ($null -eq $Report) { return '—' }
    if (-not $Report.measurable) { return 'n/a' }
    if (-not $Report.spilled) { return 'none' }
    $growth = [double]$Report.peak_shared_bytes - [double]$Report.baseline_shared_bytes
    if ($growth -lt 0) { $growth = 0 }
    $mib = [int][math]::Round($growth / 1MB)
    $secs = [double]$Report.total_spill_duration_ms / 1000.0
    # Invariant culture, as `Format-Duration` already does: on a French-locale
    # host `{0:N1}` renders 24.2 as "24,2", which in an English document reads
    # as twenty-four thousand two hundred.
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $peakGiB = (([double]$Report.peak_dedicated_bytes + $growth) / 1GB).ToString('N1', $inv)
    return ('{0} MiB / {1} / peak ~{2} GiB' -f $mib, (Format-Duration $secs), $peakGiB)
}

# --- Resolve a comma-separated -Only / -Skip token list to entries ---
#
# Accepts BOTH 1-based numbers (as printed by -List) and stable slugs. Matching
# on `Id` is exact, never prefix: `clt` and `clt_qwen3` are both real slugs, and
# a prefix match would make `-Only clt` ambiguous in a way the user could not see.
#
# An unknown token is a hard error rather than a silent no-op. `-Only lonrope`
# must not quietly select nothing and then stamp a green partial run.
function Resolve-Selection {
    param(
        [string]$Tokens,
        [object[]]$All
    )
    $picked = [System.Collections.Generic.List[object]]::new()
    $bad = @()
    foreach ($raw in ($Tokens -split ',')) {
        $tok = $raw.Trim()
        if ($tok -eq '') { continue }
        $hit = $null
        if ($tok -match '^\d+$') {
            $idx = [int]$tok
            # INDEX: guarded by the range test; -List prints these 1-based.
            if ($idx -ge 1 -and $idx -le $All.Count) { $hit = $All[$idx - 1] }
        }
        else {
            $hit = $All | Where-Object { $_.Id -eq $tok } | Select-Object -First 1
        }
        if ($null -eq $hit) { $bad += $tok }
        elseif (-not $picked.Contains($hit)) { $picked.Add($hit) }
    }
    if ($bad.Count -gt 0) {
        throw ("unknown entry: {0} — run 'scripts/resurrect.ps1 -List' for the {1} valid numbers and slugs" -f ($bad -join ', '), $All.Count)
    }
    return $picked
}

# --- -List: the number/slug map and per-entry state; runs nothing ---
if ($List) {
    $prior = Read-PriorState $mdPath
    Write-Host "`nresurrect entries ($($entries.Count)) - pass numbers or slugs to -Only / -Skip`n" -ForegroundColor Cyan
    Write-Host ('{0,3}  {1,-14} {2,-6} {3,-11} {4,-9} {5}' -f '#', 'SLUG', 'TIER', 'VERIFIED', 'WALL', 'NAME') -ForegroundColor DarkGray
    $i = 0
    foreach ($e in $entries) {
        $i++
        $tierTag = if ($e.Quick) { 'quick' } elseif ($e.FullOnly) { 'full' } else { '-' }
        $p = $prior[$e.Name]
        $date = if ($p) { $p.Date } else { 'never' }
        $wall = if ($p -and $p.Wall) { $p.Wall } else { '-' }
        Write-Host ('{0,3}  {1,-14} {2,-6} {3,-11} {4,-9} {5}' -f $i, $e.Id, $tierTag, $date, $wall, $e.Name)
    }
    Write-Host "`nexamples:  -Only longrope   |   -Only 12   |   -Only clt,sae   |   -Skip longrope" -ForegroundColor DarkGray
    Write-Host "TIER: 'quick' also runs under -Quick; 'full' runs ONLY under -Full; '-' runs by default.`n" -ForegroundColor DarkGray
    return
}

# --- -Status: report staleness and exit; runs nothing ---
if ($Status) {
    $prior = Read-PriorState $mdPath
    $today = Get-Date
    $stale = @()
    $failing = @()
    $oldestDays = -1
    $oldestName = ''
    foreach ($e in $entries) {
        $s = $prior[$e.Name]
        $date = if ($s) { $s.Date } else { 'never' }
        $parsed = $date -as [datetime]
        if ($null -eq $parsed) {
            $stale += "$($e.Name): never"
            $oldestDays = [int]::MaxValue; $oldestName = $e.Name
        }
        else {
            $days = ($today - $parsed).Days
            if ($days -gt $oldestDays) { $oldestDays = $days; $oldestName = $e.Name }
            if ($days -gt $StaleDays) { $stale += "$($e.Name): $(Format-Age $days)" }
        }
        if ($s -and $s.Outcome -match 'FAIL') { $failing += $e.Name }
    }

    Write-Host 'Oracle parity suite (scripts/resurrect.ps1):' -ForegroundColor Cyan
    if ($oldestDays -eq [int]::MaxValue) {
        Write-Host "  Oldest: never verified ($oldestName)" -ForegroundColor DarkYellow
    }
    else {
        Write-Host "  Oldest verified: $(Format-Age $oldestDays) ago ($oldestName)"
    }
    if ($failing.Count -gt 0) {
        Write-Host "  FAILING: $($failing -join ', ')" -ForegroundColor Red
    }
    if ($stale.Count -eq 0) {
        Write-Host "  All $($entries.Count) oracles verified within $StaleDays days." -ForegroundColor Green
    }
    else {
        Write-Host "  Stale (> $StaleDays days) or never run: $($stale.Count) of $($entries.Count) — run scripts/resurrect.ps1 [-Full] to refresh" -ForegroundColor DarkYellow
        foreach ($item in $stale) { Write-Host "    - $item" -ForegroundColor DarkYellow }
    }
    return
}

# --- Run path ---
Push-Location $repoRoot
try {
    $partial = $false
    if ($Only) { $selected = @(Resolve-Selection -Tokens ($Only -join ',') -All $entries); $partial = $true }
    elseif ($Quick) { $selected = @($entries | Where-Object { $_.Quick }); $tier = 'Quick' }
    elseif ($Full) { $selected = $entries; $tier = 'Full' }
    else { $selected = @($entries | Where-Object { -not $_.FullOnly }); $tier = 'Default' }

    if ($Skip) {
        $dropIds = @(Resolve-Selection -Tokens ($Skip -join ',') -All $entries | ForEach-Object { $_.Id })
        $selected = @($selected | Where-Object { $dropIds -notcontains $_.Id })
        $partial = $true
    }

    if ($selected.Count -eq 0) {
        throw 'selection is empty - nothing to run (check -Only / -Skip against -List)'
    }

    # A partial run must NOT stamp a tier name implying full coverage: reading
    # "Default" off a three-entry run later would badly misrepresent what was
    # verified. Name the entries instead, which also echoes back the slugs when
    # the run was invoked by number.
    if ($partial) {
        $tier = 'partial ({0} of {1}: {2})' -f $selected.Count, $entries.Count,
            (($selected | ForEach-Object { $_.Id }) -join ', ')
    }

    Write-Host "Resurrecting oracle suite - tier: $tier ($($selected.Count) of $($entries.Count) entries)" -ForegroundColor Cyan
    $rustc = (rustc --version)

    $resultByName = @{}
    $durationByName = @{}
    $spillByName = @{}

    # `hmn` is hypomnesis's CLI. Resolve it once: absence is a note per entry,
    # not a failure — spill sampling is instrumentation, never a gate.
    $hmn = (Get-Command hmn -ErrorAction SilentlyContinue)
    if ($null -ne $hmn -and ($SpillProbe -or ($selected | Where-Object { $_.Spill }))) {
        Write-Host "Spill sampling via $($hmn.Source)" -ForegroundColor DarkGray
    }

    # GPU identity + driver version, stamped for provenance alongside the
    # toolchain. A driver change can move GPU floating-point results, so a
    # record that pins `rustc` but not the driver can certify a configuration
    # the machine no longer runs. That is not hypothetical: the v0.1.22
    # verification moved 591.86 -> 610.88 mid-run after a `0x133_ISR_nvlddmkm`
    # bugcheck, and had the interrupted run completed instead, this file would
    # have asserted parity on a driver that was already gone.
    #
    # Read from hypomnesis >= 0.2.9's `hmn --json`, which exists because this
    # script asked for it (docs/dogfooding-feedbacks in the hypomnesis repo).
    # Absence is a note, never a gate — the same discipline the spill probe
    # already follows, so an older `hmn`, a missing `hmn`, or a non-NVIDIA
    # machine degrades to an honest "not recorded" rather than failing the run.
    $gpuStamp = $null
    if ($null -ne $hmn) {
        try {
            $raw = (& $hmn.Source --json 2>$null | Out-String)
            if (-not [string]::IsNullOrWhiteSpace($raw)) {
                $parts = @(
                    foreach ($d in @($raw | ConvertFrom-Json)) {
                        $nm = if ($d.name) { $d.name } else { "GPU $($d.index)" }
                        if ($d.driver_version) { "$nm, driver $($d.driver_version)" } else { $nm }
                    }
                )
                if ($parts.Count -gt 0) { $gpuStamp = ($parts -join '; ') }
            }
        } catch { $gpuStamp = $null }
    }
    if (-not $gpuStamp) { $gpuStamp = 'not recorded (needs `hmn` 0.2.9+ on PATH)' }
    foreach ($e in $selected) {
        $cargoArgs = @('test', '--features', $e.Features, '--release')
        if ($e.Lib) { $cargoArgs += '--lib' }
        foreach ($t in $e.Tests) { $cargoArgs += @('--test', $t) }
        $cargoArgs += '--'
        if ($e.Filter) { $cargoArgs += $e.Filter }
        $cargoArgs += @('--include-ignored', '--test-threads=1', '--nocapture')
        $skipNote = ''
        if ($e.SlowSkip -and -not $Full) {
            $cargoArgs += @('--skip', $e.SlowSkip)
            $skipNote = " (skipping '$($e.SlowSkip)')"
        }

        # WDDM spill sampling, for entries whose VRAM budget is known to be tight
        # (or everywhere under -SpillProbe). `hmn spill` samples resident
        # shared-system-memory growth while dedicated VRAM saturates, which is
        # the actual spill signal: NVML "used" pins near capacity during a spill
        # and so cannot show how far over budget the run went.
        $probe = ($e.Spill -or $SpillProbe) -and $null -ne $hmn

        # Announce sampling in the entry header, not only its absence. The
        # earlier form warned when `hmn` was missing and said nothing when
        # sampling was active, so the case worth noticing was the invisible
        # one: the only evidence was a single line printed at the very top of a
        # 40-minute run, which is not where anyone is looking when an entry
        # starts crawling.
        $probeNote = if ($probe) { ' (spill sampling)' } else { '' }

        Write-Host "`n=== $($e.Name)$skipNote$probeNote ===" -ForegroundColor Yellow
        Write-Host "cargo $($cargoArgs -join ' ')" -ForegroundColor DarkGray
        if (($e.Spill -or $SpillProbe) -and $null -eq $hmn) {
            Write-Host "  (spill sampling skipped: 'hmn' not on PATH)" -ForegroundColor DarkYellow
        }

        # True end-to-end wall-clock (model load/download + compile + run), unlike
        # cargo's own "finished in Xs" which times only the test-run phase.
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        if ($probe) {
            # --json puts the SpillReport on stdout *after* the wrapped command's
            # own output; the exit code passes through hmn unchanged.
            hmn spill --json --interval $SpillIntervalMs -- cargo @cargoArgs 2>&1 | Tee-Object -Variable teeOut
        }
        else {
            cargo @cargoArgs 2>&1 | Tee-Object -Variable teeOut
        }
        $code = $LASTEXITCODE
        $sw.Stop()
        $elapsed = $sw.Elapsed.TotalSeconds
        $durationByName[$e.Name] = $elapsed

        $skipped = @($teeOut | Select-String -SimpleMatch 'SKIP').Count -gt 0
        if ($code -ne 0) { $entryStatus = 'FAIL'; $color = 'Red' }
        elseif ($skipped) { $entryStatus = 'SKIP'; $color = 'DarkYellow' }
        else { $entryStatus = 'PASS'; $color = 'Green' }

        # Recover the SpillReport. It is emitted last, so scan from the end for
        # `"measurable"` and parse from the '{' that opens its object. Deliberately
        # NOT anchored to the start of a line: if the wrapped command's final
        # output lacks a trailing newline, hmn's JSON is appended to it (observed
        # with `cmd /c echo`), and a StartsWith('{') test would silently miss it
        # — leaving the run with no measurement and no error.
        if ($probe) {
            $report = $null
            for ($i = $teeOut.Count - 1; $i -ge 0; $i--) {
                $line = "$($teeOut[$i])"
                $marker = $line.LastIndexOf('"measurable"')
                if ($marker -lt 0) { continue }
                $start = $line.LastIndexOf('{', $marker)
                if ($start -lt 0) { continue }
                try { $report = $line.Substring($start) | ConvertFrom-Json } catch { $report = $null }
                break
            }
            if ($null -eq $report) {
                Write-Host '  (spill sampling ran but no SpillReport was parsed)' -ForegroundColor DarkYellow
            }
            $spillByName[$e.Name] = Format-Spill $report
        }

        # Prefer the measurement over the wall-clock heuristic: a long step with
        # a measured spill is stated, not guessed at.
        $slowMark = if ($spillByName.ContainsKey($e.Name) -and $spillByName[$e.Name] -ne '—') {
            " (spill $($spillByName[$e.Name]))"
        }
        elseif ($elapsed -ge $SpillWarnSeconds) { ' ⚠️ slow (VRAM spill?)' }
        else { '' }
        Write-Host "  -> $entryStatus  [$(Format-Duration $elapsed)]$slowMark" -ForegroundColor $color

        $resultByName[$e.Name] = $entryStatus
    }

    # --- Rewrite RESURRECTION.md, preserving per-entry dates for what we didn't
    #     re-verify. "Last verified" advances ONLY on a real PASS: a SKIP (model
    #     uncached) or FAIL is not a fresh verification, so it keeps the old date. ---
    $prior = Read-PriorState $mdPath
    $today = Get-Date -Format 'yyyy-MM-dd'
    $rowLines = $entries | ForEach-Object {
        $s = $resultByName[$_.Name]
        # Wall-clock is paired with "last verified": both describe the last PASS.
        $priorWall = if ($prior[$_.Name] -and $prior[$_.Name].Wall) { $prior[$_.Name].Wall } else { '—' }
        $priorSpill = if ($prior[$_.Name] -and $prior[$_.Name].Spill) { $prior[$_.Name].Spill } else { '—' }
        # Spill is paired with wall-clock and "last verified": all three describe
        # the same PASS. A step we did not sample this run keeps its prior cell.
        $spillCell = if ($spillByName.ContainsKey($_.Name)) { $spillByName[$_.Name] } else { $priorSpill }
        if ($null -ne $s) {
            $outcome = switch ($s) {
                'PASS' { '✅ PASS' }
                'SKIP' { '⏭️ SKIP (uncached)' }
                'FAIL' { '❌ FAIL' }
                default { $s }
            }
            if ($s -eq 'PASS') {
                $verified = $today
                $d = $durationByName[$_.Name]
                $wall = Format-Duration $d
                if ($d -ge $SpillWarnSeconds) { $wall += ' ⚠️' }
            }
            else {
                $verified = if ($prior[$_.Name]) { $prior[$_.Name].Date } else { 'never' }
                $wall = $priorWall
                $spillCell = $priorSpill
            }
        }
        else {
            $p = $prior[$_.Name]
            if ($p) { $verified = $p.Date; $outcome = $p.Outcome }
            else { $verified = 'never'; $outcome = '— never' }
            $wall = $priorWall
            $spillCell = $priorSpill
        }
        "| $($_.Name) | $($_.Models) | $($_.Device) | $verified | $wall | $spillCell | $outcome |"
    }

    $stampNow = Get-Date -Format 'yyyy-MM-dd HH:mm'
    $header = @(
        '# Oracle-suite resurrection log'
        ''
        'The oracle/parity tests below are `#[ignore]`d — they need cached'
        'HuggingFace models (some gated) and, for many, a CUDA GPU with ≥ 16 GiB'
        'VRAM — so GitHub CI never runs them. This file records, **per test**, when'
        'each last **passed** its oracle comparison locally.'
        ''
        '**Refresh it** with [`scripts/resurrect.ps1`](scripts/resurrect.ps1):'
        ''
        '```'
        'scripts/resurrect.ps1          # default: all but the two slow outliers (~40-50 min)'
        'scripts/resurrect.ps1 -Quick   # cheap ungated CPU encoder-parity smoke (~5 min)'
        'scripts/resurrect.ps1 -Full    # + Mistral-7B CPU forward + anacrousis 28x15 (~1.5-3 h)'
        'scripts/resurrect.ps1 -Status  # report staleness (runs nothing); -StaleDays N sets the threshold'
        '```'
        ''
        '"Last verified" = the last date this entry **passed** (a `⏭️ SKIP` /'
        '`❌ FAIL` does not advance it). `never` = not yet verified on this machine.'
        'Staleness is per-entry, so a `-Quick` run only refreshes its two rows.'
        ''
        '**Wall-clock** = end-to-end runtime of the entry on its last PASS (model'
        'load/download + compile + run, not just the `cargo test` phase). A ⚠️ flags a'
        "step slow enough (≥ $SpillWarnSeconds s) to suspect VRAM spill to shared memory"
        '(e.g. `longrope`/Phi-3.5-mini at F32 overflows a 16 GiB card).'
        ''
        '**Peak spill** = measured WDDM spill for entries sampled via `hmn spill`'
        '(hypomnesis), as *growth* of resident shared-system memory above the benign'
        'staging-heap baseline, paired with how long the spill lasted. This is the'
        'real signal: during a spill NVML `used` pins near capacity and cannot show'
        'how far over budget a run went. `none` = sampled, no spill. `—` = not'
        'sampled. `n/a` = not measurable on this platform (Linux/macOS have no'
        'shared-residency counter). Mark an entry `Spill = $true` in the script, or'
        'pass `-SpillProbe` to sample every entry.'
        ''
        "- **Last run:** $stampNow — tier **$tier**"
        "- **Toolchain:** $rustc"
        "- **GPU:** $gpuStamp"
        ''
        '| Test | Models | Device(s) | Last verified | Wall-clock | Peak spill | Outcome |'
        '|---|---|---|---|---|---|---|'
    )
    $md = ($header + $rowLines) -join "`n"
    Set-Content -Path $mdPath -Value $md -Encoding utf8
    Write-Host "`nStamped RESURRECTION.md ($tier, $stampNow)" -ForegroundColor Cyan

    # Per-run timing summary (this run's actual wall-clock, slowest first) — real
    # measurements to replace runtime guesses, and a quick spill spotter.
    if ($durationByName.Count -gt 0) {
        Write-Host "`n--- Timing summary ($tier, slowest first) ---" -ForegroundColor Cyan
        $total = 0.0
        $durationByName.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
            $total += $_.Value
            $flag = if ($_.Value -ge $SpillWarnSeconds) { '  ⚠️ VRAM spill?' } else { '' }
            Write-Host ("  {0,10}  {1}{2}" -f (Format-Duration $_.Value), $_.Key, $flag)
        }
        Write-Host ("  {0,10}  TOTAL (sum of steps; excludes inter-step overhead)" -f (Format-Duration $total)) -ForegroundColor DarkGray
    }

    $failed = @($resultByName.Values | Where-Object { $_ -eq 'FAIL' }).Count
    if ($failed -gt 0) {
        Write-Host "$failed entr$(if ($failed -eq 1) {'y'} else {'ies'}) FAILED — see output above." -ForegroundColor Red
        exit 1
    }
    Write-Host 'Done.' -ForegroundColor Green
}
finally {
    Pop-Location
}
