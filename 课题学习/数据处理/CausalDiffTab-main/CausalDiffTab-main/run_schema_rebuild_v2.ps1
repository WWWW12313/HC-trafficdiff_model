# =============================================================================
# Schema 重建 + Macro Soft Mask 修复 + 2024 数据重训练 管线
# 2026-05-16  P0→P2 正式执行脚本
#
# 变更摘要（相对旧版 macro_soft_2024）:
#   P0: data_processor.py  → IS_WEEKEND/IS_AM_PEAK/IS_PM_PEAK 替代旧时间字段
#                            is_other_vehicle 替代 is_pickup/is_van
#       _zs_expand_soft_masks.py → 加载 causal_matrix_macro_soft.npy (修复关键路径)
#       _align_causal_matrix_to_features → 名称对齐（37列矩阵 → 45列训练数据）
#   P1: prepare_dataset.py domain rules → 移除 ROAD_H3_CELL / REAL_SPEED_LIMIT / coco
#       prepare_2025_data.py → 旧字段不写入 out DataFrame
#   P2: 训练数据切换为 2024 年 NYC 事故数据，输出到 nyc_crash_2024_v2/nyc_stage1_2024_v2
#       实验 ID: macro_soft_2024_v2，λ=0.3
#
# 数据流:
#   raw_data/crash/ → prepare_2025_data --years 2024 → data/nyc_crash_2024_v2/
#   nyc_crash_2024_v2/{train,test}.csv → prepare_dataset → npy + causal_masks
#   37×37 causal_matrix_macro_soft.npy → name-aligned 45×45 → causal_masks_soft
#   train_hierarchical Stage1 (nyc_stage1_2024_v2) + Stage3 (nyc_crash_2024_v2)
#
# 使用方法：
#   cd C:\...\CausalDiffTab-main\CausalDiffTab-main
#   .\run_schema_rebuild_v2.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$PYTHON = "C:/Users/Admin/anaconda3/envs/crashgen/python.exe"
$ROOT   = $PSScriptRoot
Set-Location $ROOT

$LogFile = "$ROOT\logs\schema_rebuild_v2_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Force -Path "$ROOT\logs" | Out-Null

# 新数据集目录名
$DATANAME   = "nyc_crash_2024_v2"
$S1_DATANAME = "nyc_stage1_2024_v2"
$EXP_ID     = "macro_soft_2024_v2"

function Run-Step {
    param([string]$Label, [string]$Cmd)
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "  $Label"
    Write-Host ("=" * 70)
    Write-Host "[CMD] $Cmd"
    "=== $Label ===" | Out-File -Append -FilePath $LogFile
    "[CMD] $Cmd"     | Out-File -Append -FilePath $LogFile
    $out = Invoke-Expression "$Cmd 2>&1"
    $out | Tee-Object -Append -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Step '$Label' 退出码: $LASTEXITCODE" -ForegroundColor Red
        "[FAIL] $Label exit=$LASTEXITCODE" | Out-File -Append -FilePath $LogFile
        exit $LASTEXITCODE
    }
    Write-Host "[OK] $Label" -ForegroundColor Green
}

Write-Host "Schema Rebuild V2 管线启动 $(Get-Date)"
Write-Host "训练数据集: $DATANAME / $S1_DATANAME"
Write-Host "实验 ID:   $EXP_ID"
Write-Host "日志输出:  $LogFile"

# ─────────────────────────────────────────────────────────────
# Step 1: 构建新版宏观因果骨架 (37列, is_other_vehicle)
#         输出: configs/causal_matrix_macro_soft.{npy,csv,json}
#         注意: 已在本轮会话开头手动执行，此处幂等再次验证
# ─────────────────────────────────────────────────────────────
Run-Step "Step1: 验证/重建宏观因果骨架 (37列)" `
    "$PYTHON pipeline/build_macro_causal_skeleton.py --out_prefix configs/causal_matrix_macro_soft --verify_col_groups"

# ─────────────────────────────────────────────────────────────
# Step 2a: 更新 column_groups.json (新 schema: IS_WEEKEND 等)
#          data_processor.py 生成 processed_hierarchical.csv + column_groups.json
#          输入: 2017 v9 pristine CSV（含原始 vehicle type code, contributing factor 文本）
#          用途: 仅生成正确的 column_groups.json 供下游步骤的 schema 对齐
# ─────────────────────────────────────────────────────────────
$V9_CSV = "C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\tab-ddpm-main\data\processed\nyc_2017_pristine_v9.csv"
if (Test-Path $V9_CSV) {
    Run-Step "Step2a: 重生成 column_groups.json (新 schema)" `
        "$PYTHON src/data_processor.py --input_csv `"$V9_CSV`" --output_dir data/processed"
} else {
    Write-Host "[WARN] v9 CSV 不存在，跳过 Step2a: $V9_CSV" -ForegroundColor Yellow
    Write-Host "       请手动更新 data/processed/column_groups.json" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────
# Step 2b: 生成 2024 年训练数据 CSV (新 schema)
#          → data/nyc_crash_2024_v2/train.csv + test.csv + info.json
# ─────────────────────────────────────────────────────────────
Run-Step "Step2b: 生成 2024 训练数据 CSV (新 schema)" `
    "$PYTHON pipeline/prepare_2025_data.py --years 2024 --n_sample -1 --out_dir data/$DATANAME --train_ref data/nyc_crash_2024/train.csv"

# ─────────────────────────────────────────────────────────────
# Step 2c: CSV → npy + causal_masks (硬约束 binary mask)
#          使用 prepare_dataset.py 的 pre-split 模式
#          输出: data/nyc_crash_2024_v2/X_num_*.npy, X_cat_*.npy
#               data/nyc_stage1_2024_v2/...（自动生成 Stage1 子集）
# ─────────────────────────────────────────────────────────────
Run-Step "Step2c: 转换为 npy + 生成 binary causal_masks" `
    "$PYTHON src/prepare_dataset.py --input_csv data/$DATANAME/train.csv --input_test_csv data/$DATANAME/test.csv --notears_npy configs/causal_matrix_macro_soft.npy --full_dataname $DATANAME --stage1_dataname $S1_DATANAME"

# ─────────────────────────────────────────────────────────────
# Step 3: 展开 Macro Soft Mask
#         37×37 矩阵 → 名称对齐 → 45×45 → 写入 causal_masks_soft/
#         输出: data/nyc_crash_2024_v2/causal_masks_soft/
#              data/nyc_stage1_2024_v2/causal_masks_soft/
# ─────────────────────────────────────────────────────────────
Run-Step "Step3: 展开 Macro Soft Mask (name-aligned 37→45)" `
    "$PYTHON pipeline/_zs_expand_soft_masks.py --dataname $DATANAME --stage1_dataname $S1_DATANAME"

# ─────────────────────────────────────────────────────────────
# Step 4: 重新准备 2025 迁移测试集 (与 2024 schema 对齐)
#         --train_ref 指向新生成的 2024 训练集
# ─────────────────────────────────────────────────────────────
Run-Step "Step4: 重建 2025 迁移测试集 (对齐 2024 schema)" `
    "$PYTHON pipeline/prepare_2025_data.py --years 2025 --n_sample -1 --out_dir data/nyc_crash_2025_v2 --train_ref data/$DATANAME/train.csv"

# ─────────────────────────────────────────────────────────────
# Step 5: 训练 Stage1 (λ=0.3, soft mask, 2024 数据)
#         nyc_crash_2024_v2 → auto-map → nyc_stage1_2024_v2
# ─────────────────────────────────────────────────────────────
Run-Step "Step5: 训练 Stage1 (full tier, $EXP_ID)" `
    "$PYTHON src/train_hierarchical.py --stage 1 --tier full --experiment_id $EXP_ID --lambda_causal 0.3 --mask_subdir causal_masks_soft --device cuda:0 --dataname $DATANAME"

# ─────────────────────────────────────────────────────────────
# Step 6: 训练 Stage3 (λ=0.3, soft mask, 2024 数据)
# ─────────────────────────────────────────────────────────────
Run-Step "Step6: 训练 Stage3 (full tier, $EXP_ID)" `
    "$PYTHON src/train_hierarchical.py --stage 3 --tier full --experiment_id $EXP_ID --lambda_causal 0.3 --mask_subdir causal_masks_soft --device cuda:0 --dataname $DATANAME"

# ─────────────────────────────────────────────────────────────
# Step 7: 无条件采样 10000 条
# ─────────────────────────────────────────────────────────────
Run-Step "Step7: 无条件采样 10000 条" `
    "$PYTHON pipeline/_zs_sample_unconditional.py --experiment_id $EXP_ID --n_samples 10000 --device cuda:0"

# ─────────────────────────────────────────────────────────────
# Step 8: 全量评测 (2024 域内 + 2025 迁移)
# ─────────────────────────────────────────────────────────────
Run-Step "Step8: 全量评测 (域内 + 2025 迁移)" `
    "$PYTHON pipeline/evaluate_all.py --experiment_id $EXP_ID --transfer_test_csv data/nyc_crash_2025_v2/test.csv"

Write-Host ""
Write-Host ("=" * 70)
Write-Host "  所有步骤完成！$(Get-Date)"
Write-Host "  实验 ID:       $EXP_ID"
Write-Host "  训练数据:      data/$DATANAME  (2024 年 NYC crash)"
Write-Host "  Stage1 数据:   data/$S1_DATANAME"
Write-Host "  迁移测试集:    data/nyc_crash_2025_v2"
Write-Host "  日志:          $LogFile"
Write-Host ("=" * 70)
