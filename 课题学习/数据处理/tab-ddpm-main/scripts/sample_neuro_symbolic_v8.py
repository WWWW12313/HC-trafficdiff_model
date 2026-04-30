"""
sample_neuro_symbolic_v8.py

Neuro-Symbolic v8 sampler with strict commonsense constraints.

Pipeline:
- Stage A: Root node generation (Lat/Lon + temporal anchors)
- Stage B: Deterministic weather overwrite based on Stage A + LLM rules
- Stage C: Remaining variable generation (reference-conditioned donor transfer)
          + strict rule engine post-process:
            1) Snowplow seasonal/weather guard
            2) TOTAL_VEHICLES >= 2 if vehicle type slot1/slot2 both exist

Outputs:
- exp/nyc_crash_v8/synthetic_2017_v8.csv
- exp/nyc_crash_v8/synthetic_2017_v8.report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree


LOGGER = logging.getLogger("v8.neuro_symbolic")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def infer_time_period(hour: int) -> int:
    if 7 <= hour <= 9:
        return 1
    if 10 <= hour <= 15:
        return 2
    if 16 <= hour <= 19:
        return 3
    return 0


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def normalize_speed_tag(v: Any) -> str:
    if pd.isna(v):
        return "25 mph"
    t = str(v).strip()
    return t if t else "25 mph"


def normalize_vehicle_token(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


@dataclass
class V8SamplerConfig:
    reference_csv: str
    rules_json: str
    output_csv: str
    n_samples: int
    seed: int


class RootAnchorGenerator:
    def __init__(self, ref: pd.DataFrame, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.ref = ref.copy()

        dt = pd.to_datetime(
            self.ref["CRASH DATE"].astype(str).str.strip() + " " + self.ref["CRASH TIME"].astype(str).str.strip(),
            errors="coerce",
        )
        self.ref["_DT"] = dt
        self.ref["LATITUDE"] = safe_num(self.ref.get("LATITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref["LONGITUDE"] = safe_num(self.ref.get("LONGITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref = self.ref.dropna(subset=["LATITUDE", "LONGITUDE", "_DT"]).copy()

        self.gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=seed)
        self.gmm.fit(self.ref[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))

    def sample(self, n: int) -> pd.DataFrame:
        latlon, _ = self.gmm.sample(n)
        lat = latlon[:, 0]
        lon = latlon[:, 1]

        dt_ref = self.ref["_DT"].to_numpy()
        idx = self.rng.integers(0, len(dt_ref), size=n)
        dt = pd.Series(pd.to_datetime(dt_ref[idx]))

        jitter_min = self.rng.integers(-20, 21, size=n)
        dt = dt + pd.to_timedelta(jitter_min, unit="m")

        out = pd.DataFrame(
            {
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "CRASH_DATE_TS": (dt.astype("int64") // 10**9).astype(np.int64),
                "CRASH_TIME_MIN": (dt.dt.hour * 60 + dt.dt.minute).astype(int),
                "DAY_OF_WEEK": dt.dt.weekday.astype(int),
                "CRASH_TIME_PERIOD": dt.dt.hour.map(infer_time_period).astype(int),
                "MONTH": dt.dt.month.astype(int),
            }
        )
        out["CRASH DATE"] = dt.dt.strftime("%m/%d/%Y")
        out["CRASH TIME"] = dt.dt.strftime("%H:%M")
        return out


class DeterministicWeatherEngine:
    """Deterministic weather linkage (Stage B)."""

    MONTH_BASE_TEMP = {
        1: -1.0,
        2: 0.5,
        3: 5.0,
        4: 10.0,
        5: 16.0,
        6: 22.0,
        7: 25.0,
        8: 24.0,
        9: 20.0,
        10: 14.0,
        11: 8.0,
        12: 2.0,
    }

    MONTH_PRCP_PROB = {
        1: 0.24,
        2: 0.22,
        3: 0.26,
        4: 0.25,
        5: 0.27,
        6: 0.29,
        7: 0.31,
        8: 0.30,
        9: 0.28,
        10: 0.25,
        11: 0.24,
        12: 0.25,
    }

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    @staticmethod
    def _frac(x: np.ndarray) -> np.ndarray:
        return x - np.floor(x)

    def _deterministic_noise(self, lat: np.ndarray, lon: np.ndarray, ts: np.ndarray, offset: float) -> np.ndarray:
        key = lat * 12.9898 + lon * 78.233 + (ts / 3600.0) * 37.719 + offset + self.seed * 0.01
        raw = np.sin(key) * 43758.5453
        return self._frac(raw)

    def apply(self, anchors: pd.DataFrame) -> pd.DataFrame:
        out = anchors.copy()

        month = out["MONTH"].astype(int)
        lat = out["LATITUDE"].to_numpy(dtype=np.float64)
        lon = out["LONGITUDE"].to_numpy(dtype=np.float64)

        ts = out["CRASH_DATE_TS"].to_numpy(dtype=np.float64)
        base_temp = np.array([self.MONTH_BASE_TEMP.get(int(m), 12.0) for m in month], dtype=np.float64)
        lat_adj = (lat - 40.70) * (-4.0)
        lon_adj = (lon + 73.95) * 1.2
        diurnal = np.cos((out["CRASH_TIME_MIN"].to_numpy(dtype=np.float64) / 1440.0) * 2 * np.pi) * 2.0
        n_temp = self._deterministic_noise(lat, lon, ts, offset=1.3) - 0.5

        temp = base_temp + lat_adj + lon_adj + diurnal + n_temp * 1.8

        prcp_prob = np.array([self.MONTH_PRCP_PROB.get(int(m), 0.25) for m in month], dtype=np.float64)
        n_prcp = self._deterministic_noise(lat, lon, ts, offset=4.7)
        rain_flag = n_prcp < prcp_prob
        prcp = np.where(rain_flag, (prcp_prob - n_prcp + 1e-3) * 9.0, 0.0)

        n_wind = self._deterministic_noise(lat, lon, ts, offset=9.1)
        wind = np.clip(8.0 + np.abs(lat_adj) * 1.2 + n_wind * 14.0, 0.0, 45.0)

        # coco proxy: 1 clear, 3 cloudy, 7 rain, 15 snow
        coco = np.full(len(out), 1, dtype=int)
        coco[(prcp > 0) & (temp > 1.5)] = 7
        coco[(prcp > 0) & (temp <= 1.5)] = 15
        n_cloud = self._deterministic_noise(lat, lon, ts, offset=15.3)
        cloudy = (~rain_flag) & (n_cloud < 0.35)
        coco[cloudy] = 3

        out["CTX_TEMP"] = temp.astype(np.float32)
        out["CTX_PRCP"] = prcp.astype(np.float32)
        out["CTX_WSPD"] = wind.astype(np.float32)
        out["CTX_COCO"] = coco.astype(int)

        # mirror common weather columns for compatibility
        out["TEMP_C"] = out["CTX_TEMP"]
        out["prcp"] = out["CTX_PRCP"]
        out["WIND_SPEED_KMH"] = out["CTX_WSPD"]
        out["coco"] = out["CTX_COCO"]

        return out


class ConditionalDonorGenerator:
    """Stage C generator: transfer non-root columns from nearest reference donor."""

    def __init__(self, ref: pd.DataFrame):
        self.ref = ref.copy()
        self.ref.columns = self.ref.columns.str.strip()

        self.ref["LATITUDE"] = safe_num(self.ref.get("LATITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref["LONGITUDE"] = safe_num(self.ref.get("LONGITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref = self.ref.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

        coords = np.radians(self.ref[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        self.tree = BallTree(coords, metric="haversine")

    def fill(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = base_df.copy()

        anc_coords = np.radians(out[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        _, nn_idx = self.tree.query(anc_coords, k=1)
        donor = self.ref.iloc[nn_idx.flatten()].reset_index(drop=True)

        protected = {
            "LATITUDE",
            "LONGITUDE",
            "CRASH DATE",
            "CRASH TIME",
            "CRASH_DATE_TS",
            "CRASH_TIME_MIN",
            "DAY_OF_WEEK",
            "CRASH_TIME_PERIOD",
            "MONTH",
            "CTX_TEMP",
            "CTX_PRCP",
            "CTX_WSPD",
            "CTX_COCO",
            "TEMP_C",
            "prcp",
            "WIND_SPEED_KMH",
            "coco",
        }

        for c in donor.columns:
            if c in protected:
                continue
            out[c] = donor[c].to_numpy(copy=False)

        # fill route context if missing in source
        if "OSM_SPEED_TAG" in out.columns:
            out["OSM_SPEED_TAG"] = out["OSM_SPEED_TAG"].map(normalize_speed_tag)

        return out


class RuleEngine:
    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules

    @staticmethod
    def _vehicle_columns(df: pd.DataFrame) -> List[str]:
        candidates = [
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
            "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5",
            "VEHICLE_TYPE_1",
            "VEHICLE_TYPE_2",
            "VEHICLE_TYPE_3",
            "VEHICLE_TYPE_4",
            "VEHICLE_TYPE_5",
        ]
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _valid_vehicle_present(s: pd.Series) -> pd.Series:
        txt = s.fillna("").astype(str).str.strip().str.lower()
        return (~txt.isin(["", "nan", "none", "null", "unknown", "unspecified"]))

    def enforce_snowplow_rule(self, df: pd.DataFrame, fallback_pool: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, int]:
        out = df.copy()
        veh_cols = self._vehicle_columns(out)
        if not veh_cols:
            return out, 0

        tokens = ["snow plow", "snowplow", "plow"]
        month = out["MONTH"] if "MONTH" in out.columns else pd.Series(1, index=out.index)
        temp = safe_num(out.get("CTX_TEMP", pd.Series(20.0, index=out.index)), default=20.0)
        prcp = safe_num(out.get("CTX_PRCP", pd.Series(0.0, index=out.index)), default=0.0)
        coco = safe_num(out.get("CTX_COCO", pd.Series(1, index=out.index)), default=1).astype(int)

        winter = month.astype(int).isin([12, 1, 2])
        snowy = coco.isin([15, 16]) | ((temp <= 2.0) & (prcp > 0))
        allowed = winter | snowy

        replaced = 0
        fallback_vals = (
            fallback_pool.dropna().astype(str).tolist() if fallback_pool is not None else ["Sedan", "SUV", "Taxi", "Pick-up Truck"]
        )
        if not fallback_vals:
            fallback_vals = ["Sedan", "SUV"]

        for c in veh_cols:
            s = out[c].fillna("").astype(str)
            is_snow_related = s.str.lower().map(lambda x: any(t in x for t in tokens))
            bad = is_snow_related & (~allowed)
            bad_count = int(bad.sum())
            if bad_count > 0:
                repl = np.random.choice(fallback_vals, size=bad_count, replace=True)
                out.loc[bad, c] = repl
                replaced += bad_count

        return out, replaced

    def enforce_multi_vehicle_math(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        out = df.copy()

        c1 = "VEHICLE TYPE CODE 1" if "VEHICLE TYPE CODE 1" in out.columns else "VEHICLE_TYPE_1"
        c2 = "VEHICLE TYPE CODE 2" if "VEHICLE TYPE CODE 2" in out.columns else "VEHICLE_TYPE_2"
        if c1 not in out.columns or c2 not in out.columns:
            return out, 0

        both_present = self._valid_vehicle_present(out[c1]) & self._valid_vehicle_present(out[c2])

        if "TOTAL_VEHICLES" not in out.columns:
            out["TOTAL_VEHICLES"] = 1

        total = safe_num(out["TOTAL_VEHICLES"], default=1.0).round().astype(int)
        need_fix = both_present & (total < 2)
        fix_count = int(need_fix.sum())

        if fix_count > 0:
            total.loc[need_fix] = 2
            out["TOTAL_VEHICLES"] = total.astype(int)

        if "IS_MULTI_VEHICLE" in out.columns:
            out["IS_MULTI_VEHICLE"] = (safe_num(out["TOTAL_VEHICLES"], default=1.0) >= 2).astype(int)

        return out, fix_count

    def enforce_casualty_sum_logic(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        out = df.copy()
        req = [
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
            "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST KILLED",
        ]
        for c in req:
            if c not in out.columns:
                out[c] = 0

        ped_inj = safe_num(out["NUMBER OF PEDESTRIANS INJURED"], default=0.0).clip(lower=0).round().astype(int)
        cyc_inj = safe_num(out["NUMBER OF CYCLIST INJURED"], default=0.0).clip(lower=0).round().astype(int)
        mot_inj = safe_num(out["NUMBER OF MOTORIST INJURED"], default=0.0).clip(lower=0).round().astype(int)

        ped_kil = safe_num(out["NUMBER OF PEDESTRIANS KILLED"], default=0.0).clip(lower=0).round().astype(int)
        cyc_kil = safe_num(out["NUMBER OF CYCLIST KILLED"], default=0.0).clip(lower=0).round().astype(int)
        mot_kil = safe_num(out["NUMBER OF MOTORIST KILLED"], default=0.0).clip(lower=0).round().astype(int)

        expected_inj = ped_inj + cyc_inj + mot_inj
        expected_kil = ped_kil + cyc_kil + mot_kil

        old_inj = safe_num(out.get("NUMBER OF PERSONS INJURED", pd.Series(0, index=out.index)), default=0.0).round().astype(int)
        old_kil = safe_num(out.get("NUMBER OF PERSONS KILLED", pd.Series(0, index=out.index)), default=0.0).round().astype(int)

        changes = int(((old_inj != expected_inj) | (old_kil != expected_kil)).sum())

        out["NUMBER OF PERSONS INJURED"] = expected_inj.astype(int)
        out["NUMBER OF PERSONS KILLED"] = expected_kil.astype(int)
        return out, changes


def load_rules(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rules JSON not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def run(cfg: V8SamplerConfig) -> None:
    LOGGER.info("Loading reference CSV: %s", cfg.reference_csv)
    ref = pd.read_csv(cfg.reference_csv)
    ref.columns = ref.columns.str.strip()

    rules = load_rules(cfg.rules_json)
    LOGGER.info("Loaded rules JSON: %s", cfg.rules_json)

    # Stage A
    LOGGER.info("Stage A: generating root anchors (n=%d)", cfg.n_samples)
    root_gen = RootAnchorGenerator(ref, seed=cfg.seed)
    roots = root_gen.sample(cfg.n_samples)

    # Stage B
    LOGGER.info("Stage B: deterministic weather overwrite")
    weather_engine = DeterministicWeatherEngine(seed=cfg.seed)
    stage_b = weather_engine.apply(roots)

    # Stage C
    LOGGER.info("Stage C: conditional donor generation for non-root variables")
    donor_gen = ConditionalDonorGenerator(ref)
    synth = donor_gen.fill(stage_b)

    # Rule Engine
    LOGGER.info("Applying strict rule engine constraints")
    fallback_vehicle_pool = None
    if "VEHICLE TYPE CODE 1" in ref.columns:
        fallback_vehicle_pool = ref["VEHICLE TYPE CODE 1"]

    engine = RuleEngine(rules)
    synth, n_snow_fix = engine.enforce_snowplow_rule(synth, fallback_pool=fallback_vehicle_pool)
    synth, n_multi_fix = engine.enforce_multi_vehicle_math(synth)
    synth, n_casualty_fix = engine.enforce_casualty_sum_logic(synth)

    # Keep IS_* columns at end if present
    is_cols = [c for c in synth.columns if c.startswith("IS_")]
    other_cols = [c for c in synth.columns if not c.startswith("IS_")]
    synth = synth[other_cols + is_cols]

    out_csv = Path(cfg.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {
        "version": "v8",
        "output_csv": out_csv.as_posix(),
        "rows": int(len(synth)),
        "cols": int(synth.shape[1]),
        "rules_json": cfg.rules_json,
        "fix_stats": {
            "snowplow_replacements": int(n_snow_fix),
            "multi_vehicle_total_repairs": int(n_multi_fix),
            "casualty_sum_repairs": int(n_casualty_fix),
        },
    }
    report_path = out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved synthetic v8 CSV: %s", out_csv.as_posix())
    LOGGER.info("Saved report: %s", report_path.as_posix())
    LOGGER.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-Symbolic v8 sampler")
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--rules_json", type=str, default="data/nyc_crash_v8/llm_causal_rules.json")
    parser.add_argument("--output_csv", type=str, default="exp/nyc_crash_v8/synthetic_2017_v8.csv")
    parser.add_argument("--n_samples", type=int, default=159992)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cfg = V8SamplerConfig(
        reference_csv=args.reference_csv,
        rules_json=args.rules_json,
        output_csv=args.output_csv,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
