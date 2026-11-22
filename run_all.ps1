# ZebraSight: every remaining step, in order. Resumable: a step whose outputs already exist is skipped.
# Each step logs to logs\<step>.log (stdout) and logs\<step>.err (stderr, warnings). Stops on the first failure.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File C:\zebrasight\run_all.ps1
param([switch]$Force)   # -Force: rerun every step even if its outputs exist

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location $Root

Write-Host @"
ZebraSight run_all  (estimated time per step, RTX 3050 laptop; skipped steps take ~0 s)
  1 c9 percepts complete check / finish ............... ~1 min  (GPU)
  2 MegaDescriptor sanity check on ORIGINAL crops ...... ~1 min  (GPU; embeddings cached)
  3a features c8+c9 (n_jobs 6) ........................ ~4 min  (CPU)
  3b MegaDescriptor embeddings c8+c9 ................... ~12 min (GPU)
  4 metrics c0, c8, c9 (200 bootstraps, + OWLv2 face) .. ~3-4 min each, ~10 min total (CPU)
  5 figures ............................................ ~1 min
  6 app_data\ + app_data.zip (real data) ............... ~2 min
  total from scratch ~30 min
"@

function Invoke-Step([string]$Name, [string[]]$PyArgs, [string[]]$Outputs) {
    if (-not $Force -and $Outputs.Count -gt 0 -and -not ($Outputs | Where-Object { -not (Test-Path (Join-Path $Root $_)) })) {
        Write-Host ("[{0}] SKIP {1} (outputs exist)" -f (Get-Date -Format HH:mm:ss), $Name)
        return
    }
    $log = Join-Path $Logs "$Name.log"
    $err = Join-Path $Logs "$Name.err"
    Write-Host ("[{0}] START {1}  (log: logs\{1}.log)" -f (Get-Date -Format HH:mm:ss), $Name)
    $t0 = Get-Date
    $p = Start-Process -FilePath $Py -ArgumentList $PyArgs -WorkingDirectory $Root -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    if ($p.ExitCode -ne 0) {
        Write-Host ""
        Write-Host ("STEP FAILED: {0} (exit code {1}) after {2} s" -f $Name, $p.ExitCode, $dt) -ForegroundColor Red
        Write-Host "Last lines of logs\$Name.log:"; Get-Content $log -Tail 8
        Write-Host "Last lines of logs\$Name.err:"; Get-Content $err -Tail 15
        Write-Host "Fix the problem, then run this script again: finished steps are skipped." -ForegroundColor Yellow
        exit 1
    }
    Get-Content $log -Tail 4
    Write-Host ("[{0}] DONE  {1} in {2} s" -f (Get-Date -Format HH:mm:ss), $Name, $dt) -ForegroundColor Green
}

# 1 finish any missing c9 percepts (the store driver itself skips rows already computed)
$c = Start-Process -FilePath $Py -ArgumentList "scripts\check_store.py", "cache\percepts_50.zarr", "c9" -WorkingDirectory $Root -NoNewWindow -Wait -PassThru
if ($Force -or $c.ExitCode -ne 0) {
    Invoke-Step "1_percepts_c9" @("-u", "-m", "zebrasight.pipeline.store", "data\regions_50.parquet", "cache\percepts_50.zarr",
        "--conds", "c9", "--regions-used", "flank,face") @()
} else { Write-Host ("[{0}] SKIP 1_percepts_c9 (store complete)" -f (Get-Date -Format HH:mm:ss)) }

# 2 MegaDescriptor sanity check on ORIGINAL crops (exit code 2 = flank near chance: crop/split problem -> stop)
Invoke-Step "2_sanity_originals" @("-u", "-m", "zebrasight.metrics.sanity") @("outputs\metrics\originals.csv")
if (Test-Path (Join-Path $Logs "2_sanity_originals.log")) { Get-Content (Join-Path $Logs "2_sanity_originals.log") | Select-String "AUC" }

# 3 features + MegaDescriptor embeddings for c8 and c9
Invoke-Step "3a_features_c8c9" @("-u", "-m", "zebrasight.features.run", "cache\percepts_50.zarr", "outputs\features\features_c8c9_v2.parquet",
    "--regions", "flank,face", "--conds", "c8,c9", "--n-jobs", "6") @("outputs\features\features_c8c9_v2.parquet")
Invoke-Step "3b_megadesc_c8c9" @("-u", "-m", "zebrasight.metrics.megadesc", "percepts", "cache\percepts_50.zarr",
    "outputs\features\megadesc_c8c9.parquet", "--conds", "c8,c9") @("outputs\features\megadesc_c8c9.parquet")
Invoke-Step "3c_megadesc_c0" @("-u", "-m", "zebrasight.metrics.megadesc", "percepts", "cache\percepts_50.zarr",
    "outputs\features\megadesc_c0.parquet", "--conds", "c0") @("outputs\features\megadesc_c0.parquet")

# 3d HotSpotter matcher + registered layout on the percepts (amendments A1, A4), all three lights (~3 min, CPU)
Invoke-Step "3d_hotspotter_percepts" @("-u", "-m", "zebrasight.metrics.hotspotter", "percepts", "cache\percepts_50.zarr",
    "outputs\features\hotspotter_percepts.parquet", "--conds", "c0,c8,c9", "--n-jobs", "6") @("outputs\features\hotspotter_percepts.parquet")

# 4 metrics (200 bootstraps; the OWLv2-only face check is the pseudo-region face_owlv2 in every run)
Invoke-Step "4_metrics_c0" @("-u", "-m", "zebrasight.metrics.run", "outputs\features\features_c0_v2.parquet", "outputs\metrics",
    "--cond", "c0", "--embeddings", "outputs\features\megadesc_c0.parquet", "--pairs", "outputs\features\hotspotter_percepts.parquet", "--n-boot", "200") `
    @("outputs\metrics\metrics_c0.parquet", "outputs\metrics\boot_c0.npz", "outputs\metrics\family_c0.parquet")
foreach ($cond in "c8", "c9") {
    Invoke-Step "4_metrics_$cond" @("-u", "-m", "zebrasight.metrics.run", "outputs\features\features_c8c9_v2.parquet", "outputs\metrics",
        "--cond", $cond, "--embeddings", "outputs\features\megadesc_c8c9.parquet", "--pairs", "outputs\features\hotspotter_percepts.parquet", "--n-boot", "200") `
        @("outputs\metrics\metrics_$cond.parquet", "outputs\metrics\boot_$cond.npz", "outputs\metrics\family_$cond.parquet")
}

# 5 figures (cheap: always rerun so they match the latest metrics)
Invoke-Step "5_figures" @("-u", "-m", "zebrasight.metrics.figures") @()
Invoke-Step "5b_stability" @("-u", "-m", "zebrasight.plots.stability") @()

# 6 app data from real results (always rerun) -> app_data\ and app_data.zip
Invoke-Step "6_app_export" @("-u", "-m", "zebrasight.app_export") @()

Write-Host ""
Write-Host "ALL STEPS DONE. Results: app_data\results.json, app_data.zip, outputs\figures\" -ForegroundColor Green
Write-Host "Check app_data\results.json has  ""status"": ""final""  before the demo."
