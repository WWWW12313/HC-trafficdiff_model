"""
llm_causal_prior_v8.py

LLM-Driven causal prior extraction for v8 (Neuro-Symbolic pipeline).
- Builds schema-aware prompt from dataset columns.
- Queries a mockable LLM client.
- Validates and saves deterministic rules + adjacency constraints.

Output:
    data/nyc_crash_v8/llm_causal_rules.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


LOGGER = logging.getLogger("v8.llm_prior")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class LLMClient(Protocol):
    """Interface for swappable LLM backends."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class MockLLMClient:
    """Deterministic local fallback client for offline/reproducible runs."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        rules = {
            "version": "v8",
            "rule_source": "mock_llm",
            "root_nodes": [
                "LATITUDE",
                "LONGITUDE",
                "CRASH_DATE_TS",
                "CRASH_TIME_MIN",
                "DAY_OF_WEEK",
                "CRASH_TIME_PERIOD",
            ],
            "deterministic_rules": [
                {
                    "id": "R1_SPATIOTEMP_TO_WEATHER",
                    "description": "Weather columns are deterministic function of (lat, lon, time); never sampled independently.",
                    "type": "function_overwrite",
                    "inputs": ["LATITUDE", "LONGITUDE", "CRASH_DATE_TS", "CRASH_TIME_MIN"],
                    "outputs": ["CTX_TEMP", "CTX_PRCP", "CTX_WSPD", "CTX_COCO"],
                    "strict": True,
                },
                {
                    "id": "R2_SNOWPLOW_SEASONAL",
                    "description": "Snow-plow-like vehicles are valid only in winter-like or snowy conditions.",
                    "type": "guard",
                    "if_any_vehicle_contains": ["snow plow", "snowplow", "plow"],
                    "requires_any": ["MONTH in [12,1,2]", "CTX_COCO in [15,16]", "CTX_TEMP <= 2.0", "CTX_PRCP > 0.0"],
                    "strict": True,
                },
                {
                    "id": "R3_MULTI_VEHICLE_MATH",
                    "description": "If vehicle slot 1 and 2 both exist, TOTAL_VEHICLES must be >= 2.",
                    "type": "algebraic",
                    "if_non_null": ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"],
                    "enforce": "TOTAL_VEHICLES >= 2",
                    "strict": True,
                },
            ],
            "adjacency_mask": {
                "allow_edges": [
                    ["LATITUDE", "CTX_TEMP"],
                    ["LONGITUDE", "CTX_TEMP"],
                    ["CRASH_DATE_TS", "CTX_TEMP"],
                    ["CRASH_DATE_TS", "CTX_PRCP"],
                    ["CRASH_TIME_MIN", "CTX_COCO"],
                    ["CTX_COCO", "VEHICLE TYPE CODE 1"],
                    ["CTX_TEMP", "VEHICLE TYPE CODE 1"],
                    ["CTX_PRCP", "VEHICLE TYPE CODE 1"],
                    ["VEHICLE TYPE CODE 1", "TOTAL_VEHICLES"],
                    ["VEHICLE TYPE CODE 2", "TOTAL_VEHICLES"],
                ],
                "forbid_edges": [
                    ["CTX_TEMP", "LATITUDE"],
                    ["CTX_PRCP", "LONGITUDE"],
                    ["VEHICLE TYPE CODE 1", "CRASH_DATE_TS"],
                    ["TOTAL_VEHICLES", "VEHICLE TYPE CODE 1"],
                ],
            },
        }
        return json.dumps(rules, ensure_ascii=False, indent=2)


@dataclass
class OpenAICompatLLMClient:
    """Optional OpenAI-compatible backend. Falls back to mock if unavailable."""

    model: str = "gpt-4o-mini"

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            LOGGER.warning("OpenAI SDK unavailable (%s), fallback to mock rules.", e)
            return MockLLMClient().generate(prompt)

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            LOGGER.warning("OPENAI_API_KEY not found, fallback to mock rules.")
            return MockLLMClient().generate(prompt)

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a causal discovery assistant. Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content


def build_schema_prompt(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()

    prompt = {
        "task": "Generate causal priors and adjacency constraints for NYC crash table synthesis.",
        "requirements": {
            "strict_rules": [
                "Weather must be deterministic from (lat, lon, time), not random.",
                "Snow-plow-like vehicle type only allowed in winter/snow-compatible conditions.",
                "If vehicle type slot 1 and 2 both non-empty, TOTAL_VEHICLES >= 2.",
            ],
            "output_json_schema": {
                "version": "string",
                "rule_source": "string",
                "root_nodes": ["string"],
                "deterministic_rules": ["object"],
                "adjacency_mask": {
                    "allow_edges": [["src", "dst"]],
                    "forbid_edges": [["src", "dst"]],
                },
            },
        },
        "available_columns": cols,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def validate_rules(payload: Dict[str, Any]) -> None:
    required_top = ["version", "root_nodes", "deterministic_rules", "adjacency_mask"]
    for k in required_top:
        if k not in payload:
            raise ValueError(f"Missing key in rules JSON: {k}")

    adj = payload.get("adjacency_mask", {})
    if "allow_edges" not in adj or "forbid_edges" not in adj:
        raise ValueError("adjacency_mask must contain allow_edges and forbid_edges")

    if not isinstance(payload.get("root_nodes", []), list):
        raise ValueError("root_nodes must be a list")


def resolve_client(mode: str, model: str) -> LLMClient:
    if mode == "openai":
        LOGGER.info("Using OpenAI-compatible LLM backend: model=%s", model)
        return OpenAICompatLLMClient(model=model)
    LOGGER.info("Using mock LLM backend")
    return MockLLMClient()


def run(input_csv: str, output_json: str, llm_mode: str, model: str) -> None:
    LOGGER.info("Loading schema source CSV: %s", input_csv)
    df = pd.read_csv(input_csv, nrows=5000)
    df.columns = df.columns.str.strip()

    prompt = build_schema_prompt(df)
    LOGGER.info("Prompt built with %d columns", len(df.columns))

    client = resolve_client(llm_mode, model)
    raw = client.generate(prompt)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        LOGGER.error("LLM output is not valid JSON: %s", e)
        raise

    validate_rules(payload)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved v8 LLM causal rules to: %s", out_path.as_posix())
    LOGGER.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LLM causal priors for v8")
    parser.add_argument("--input_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--output_json", type=str, default="data/nyc_crash_v8/llm_causal_rules.json")
    parser.add_argument("--llm_mode", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    run(args.input_csv, args.output_json, args.llm_mode, args.model)


if __name__ == "__main__":
    main()
