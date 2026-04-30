param(
    [string]$PythonExe = "C:/Users/Admin/anaconda3/envs/crashgen/python.exe",
    [string]$PrimaryModel = "models/gemini-2.5-flash",
    [string]$CompareModel = "models/gemini-2.5-pro",
    [string]$OutputRoot = "exp/nyc_crash_v8_ablation_lowquota",
    [int]$Seed = 42,
    [int]$NSamples = 159992,
    [int]$GeminiMaxRows = 60,
    [double]$GeminiSleepS = 1.5
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $env:GEMINI_API_KEY) {
    $env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
}

if (-not $env:GEMINI_API_KEY) {
    throw "GEMINI_API_KEY is empty. Set user env first."
}

$baseDir = Join-Path $OutputRoot "base"
$flashDir = Join-Path $OutputRoot "flash"
$proDir = Join-Path $OutputRoot "pro"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null
New-Item -ItemType Directory -Force -Path $flashDir | Out-Null
New-Item -ItemType Directory -Force -Path $proDir | Out-Null

Write-Host "[1/5] Build deterministic base (free+hard)"
& $PythonExe scripts/v8_ablation_sampler.py --mode free --output_dir $baseDir --n_samples $NSamples --seed $Seed --weather_source deterministic --osm_source donor
& $PythonExe scripts/v8_ablation_sampler.py --mode hard --output_dir $baseDir --n_samples $NSamples --seed $Seed --weather_source deterministic --osm_source donor

Write-Host "[2/5] Copy base files to flash/pro"
Copy-Item "$baseDir/synthetic_free.csv","$baseDir/synthetic_free_meta.json","$baseDir/synthetic_hard.csv","$baseDir/synthetic_hard_meta.json" -Destination $flashDir -Force
Copy-Item "$baseDir/synthetic_free.csv","$baseDir/synthetic_free_meta.json","$baseDir/synthetic_hard.csv","$baseDir/synthetic_hard_meta.json" -Destination $proDir -Force

Write-Host "[3/5] Run soft with primary model: $PrimaryModel"
& $PythonExe scripts/v8_ablation_sampler.py --mode soft --output_dir $flashDir --n_samples $NSamples --seed $Seed --penalty_backend gemini --gemini_model $PrimaryModel --quota_profile low --gemini_max_rows $GeminiMaxRows --gemini_sleep_s $GeminiSleepS
& $PythonExe scripts/v8_ablation_eval.py --base_dir $flashDir --out_md "$flashDir/v8_ablation_report.md" --out_json "$flashDir/v8_ablation_metrics.json"

Write-Host "[4/5] Run soft with compare model: $CompareModel"
& $PythonExe scripts/v8_ablation_sampler.py --mode soft --output_dir $proDir --n_samples $NSamples --seed $Seed --penalty_backend gemini --gemini_model $CompareModel --quota_profile low --gemini_max_rows $GeminiMaxRows --gemini_sleep_s $GeminiSleepS
& $PythonExe scripts/v8_ablation_eval.py --base_dir $proDir --out_md "$proDir/v8_ablation_report.md" --out_json "$proDir/v8_ablation_metrics.json"

Write-Host "[5/5] Save run manifest"
$manifest = [ordered]@{
    created_at = (Get-Date).ToString("s")
    python_exe = $PythonExe
    seed = $Seed
    n_samples = $NSamples
    quota_profile = "low"
    gemini_max_rows = $GeminiMaxRows
    gemini_sleep_s = $GeminiSleepS
    primary_model = $PrimaryModel
    compare_model = $CompareModel
    output_root = $OutputRoot
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OutputRoot "run_manifest.json") -Encoding UTF8

Write-Host "Done. Outputs:"
Write-Host "  $flashDir"
Write-Host "  $proDir"
