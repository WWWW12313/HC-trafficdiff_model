param(
    [string]$RepoRoot = "",
    [string]$PythonCmd = "conda run -n crashgen python",
    [switch]$SkipBaselines,
    [switch]$SkipV8,
    [switch]$LLMOpenAI,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

Set-Location $RepoRoot

$python = $PythonCmd
$v9Csv = "nyc_2017_pristine_v9.csv"

$v6Data = "data/nyc_crash_v3_v9"
$v7Data = "data/nyc_crash_v7_v9"
$v6Exp = "exp/nyc_crash_v3/causal_m4_v6_v9"
$v7Exp = "exp/nyc_crash_v7/causal_m4_v7_v9"
$baselineExp = "exp/nyc_crash_v3_v9_baselines"
$v8Exp = "exp/nyc_crash_v8_ablation_v9_rerun"

$commands = @(
    "$python scripts/prepare_data_v3.py --csv $v9Csv --output $v6Data --seed 42",
    "$python scripts/prepare_data_v7.py --input_csv $v9Csv --output_dir $v7Data --seed 42",

    "$python train_causal_v6.py --data $v6Data --output $v6Exp --mode balanced",
    "$python scripts/postprocess_synthetic_v3.py --parent_dir $v6Exp --data_dir $v6Data --output $v6Exp/synthetic_complete_v9.csv --skip_api --pristine_csv $v9Csv",
    "$python scripts/evaluate_v3.py --real_dir $v6Data --syn_dir $v6Exp --model_name CausalDDPM_v6_v9 --output_dir exp/nyc_crash_v3",

    "$python train_causal_v7.py --data $v7Data --output $v7Exp --mode balanced",
    "$python scripts/sample_two_stage_v7.py --model_dir $v7Exp --data_dir $v7Data --reference_csv $v9Csv --output_csv $v7Exp/synthetic_2017_v7_v9.csv --seed 42",
    "$python scripts/evaluate_v7.py --real_csv $v9Csv --syn_csv $v7Exp/synthetic_2017_v7_v9.csv --out_json $v7Exp/eval_v7_v9.json"
)

if (-not $SkipBaselines) {
    $commands += "$python run_baselines_v3.py --model all --real_data $v6Data --exp_base $baselineExp"
}

if (-not $SkipV8) {
    $commands += "$python scripts/v8_ablation_sampler.py --mode free --reference_csv $v9Csv --output_dir $v8Exp --seed 42"
    $commands += "$python scripts/v8_ablation_sampler.py --mode hard --reference_csv $v9Csv --output_dir $v8Exp --seed 42"
    $commands += "$python scripts/v8_ablation_sampler.py --mode soft --reference_csv $v9Csv --output_dir $v8Exp --seed 42 --penalty_backend gemini --quota_profile low --gemini_max_rows 50"
    $commands += "$python scripts/v8_ablation_eval.py --real_csv $v9Csv --base_dir $v8Exp --out_md $v8Exp/v8_ablation_report.md --out_json $v8Exp/v8_ablation_metrics.json"
}

if ($LLMOpenAI) {
    $commands += "$python scripts/llm_physics_dag_v9.py --input_csv $v9Csv --llm_mode openai --model gpt-4o-mini --output_json exp/nyc_crash_v9_llm/physics_dag_v9_openai.json"
} else {
    $commands += "$python scripts/llm_physics_dag_v9.py --input_csv $v9Csv --llm_mode mock --output_json exp/nyc_crash_v9_llm/physics_dag_v9_mock.json"
}

Write-Host "RepoRoot: $RepoRoot"
Write-Host "PythonCmd: $python"
Write-Host "Total steps: $($commands.Count)"

$index = 1
foreach ($cmd in $commands) {
    Write-Host "[$index/$($commands.Count)] $cmd"
    if (-not $DryRun) {
        Invoke-Expression $cmd
    }
    $index++
}

Write-Host "v9 full rerun pipeline completed."
