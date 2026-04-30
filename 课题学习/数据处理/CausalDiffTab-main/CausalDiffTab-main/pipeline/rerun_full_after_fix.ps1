$proj = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $proj

$logDir = Join-Path $proj "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format yyyyMMdd_HHmmss
$log = Join-Path $logDir "full_tier_all_models_rerun_after_fix_$ts.log"

"FULL tier rerun(after fix) started $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8

$models = @('baseline_tabddpm','ablation_no_causal','ablation_no_hierarchy','ours_full_model')
$py = "C:\Users\Admin\anaconda3\envs\crashgen\python.exe"
$runScript = Join-Path $proj "pipeline\run_all_experiments.py"

if (-not (Test-Path $py)) {
  throw "Python not found: $py"
}
if (-not (Test-Path $runScript)) {
  throw "run_all_experiments.py not found: $runScript"
}

foreach ($m in $models) {
  "=== START $m $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
  & $py $runScript --model $m --tier full 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) {
    "[FAILED] $m exit=$LASTEXITCODE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    break
  }
  "[DONE] $m $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
}

"FULL tier rerun(after fix) finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
Write-Output "LOG_PATH=$log"
