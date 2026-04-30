# Low-Quota Reproducible Preset (V8 Ablation)

This preset is designed for long-term reproducibility under strict Gemini quota.

## What makes it reproducible

- Fixed seed (`--seed 42` by default)
- Deterministic context (`weather_source=deterministic`, `osm_source=donor`)
- Fixed low-quota profile (`--quota_profile low`)
- Explicit model IDs and run manifest output (`run_manifest.json`)

## Low-quota profile defaults in sampler

When `--quota_profile low` is enabled, the sampler auto-adjusts defaults unless overridden explicitly:

- `weather_source=deterministic`
- `osm_source=donor`
- `gemini_max_rows=60`
- `gemini_sleep_s=1.5`
- `gemini_temperature=0.0`
- `gemini_retries=4`
- `gemini_backoff_base_s=2.0`
- `gemini_backoff_cap_s=45.0`

## One-click run (PowerShell)

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v8_ablation_low_quota_repro.ps1
```

Custom models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v8_ablation_low_quota_repro.ps1 `
  -PrimaryModel "models/gemini-2.5-flash" `
  -CompareModel "models/gemini-2.5-pro"
```

## Output structure

- `exp/nyc_crash_v8_ablation_lowquota/base/`
- `exp/nyc_crash_v8_ablation_lowquota/flash/`
- `exp/nyc_crash_v8_ablation_lowquota/pro/`
- `exp/nyc_crash_v8_ablation_lowquota/run_manifest.json`

## Note on 429 errors

Even with this preset, severe account-level quota exhaustion can still cause fallback penalties. The profile reduces burst pressure and improves retry behavior, but cannot bypass provider-side quota limits.
