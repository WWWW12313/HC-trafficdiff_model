from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompts(columns: List[str], cfg: Dict[str, Any]) -> Dict[str, str]:
    llm_cfg = cfg["llm_causal_discovery"]

    system_prompt = llm_cfg.get("system_prompt_template", "TODO: fill system prompt")
    few_shot = llm_cfg.get("few_shot_template", "TODO: fill few-shot examples")

    # Keep explicit placeholders so prompt engineering can be iterated safely.
    user_prompt = (
        "[Variables]\n"
        + "\n".join([f"- {c}" for c in columns])
        + "\n\n[Instruction]\n"
        + "Return JSON with keys: reasoning, causal_edges[{source,target}]."
    )
    return {
        "system_prompt": system_prompt,
        "few_shot": few_shot,
        "user_prompt": user_prompt,
    }


def mock_llm_response(columns: List[str]) -> Dict[str, Any]:
    # Deterministic fallback DAG edges for offline development.
    edges: List[Dict[str, str]] = []
    for i in range(len(columns) - 1):
        edges.append({"source": columns[i], "target": columns[i + 1]})
    return {
        "reasoning": "mock mode: generated a simple acyclic chain over input columns",
        "causal_edges": edges,
    }


def call_llm_with_retry(columns: List[str], prompts: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = cfg["llm_causal_discovery"]
    mode = str(llm_cfg.get("mode", "mock")).lower()
    max_retries = int(llm_cfg.get("max_retries", 5))
    base_backoff = int(llm_cfg.get("initial_backoff_seconds", 4))

    if mode == "mock":
        return mock_llm_response(columns)

    # TODO: replace with Gemini/OpenAI API call implementation.
    # This block intentionally keeps retry/backoff skeleton for future online integration.
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            raise RuntimeError("LLM API client is not implemented yet. Fill provider client in pipeline/01.")
        except Exception as exc:
            last_error = exc
            sleep_s = base_backoff * (2 ** attempt)
            print(f"[pipeline/01] LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(4)
                time.sleep(max(0, sleep_s - 4))

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def build_adjacency_matrix(columns: List[str], edges: List[Dict[str, str]]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(columns)}
    mat = np.zeros((len(columns), len(columns)), dtype=np.int8)
    for edge in edges:
        s = edge.get("source")
        t = edge.get("target")
        if s in idx and t in idx and s != t:
            mat[idx[s], idx[t]] = 1
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 01: LLM causal discovery -> causal matrix")
    parser.add_argument("--config", type=str, default="configs/v9_experiment.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(repo_root / args.config)

    columns = list(cfg["features"]["columns"])
    prompts = build_prompts(columns, cfg)
    payload = call_llm_with_retry(columns, prompts, cfg)

    edges = payload.get("causal_edges", [])
    mat = build_adjacency_matrix(columns, edges)

    llm_cfg = cfg["llm_causal_discovery"]
    npy_path = repo_root / llm_cfg.get("output_npy", "exp/causal_matrix_v9.npy")
    json_path = repo_root / llm_cfg.get("output_json", "exp/causal_matrix_v9.json")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(npy_path, mat)

    log_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": llm_cfg.get("mode", "mock"),
        "provider": llm_cfg.get("provider", "gemini"),
        "model": llm_cfg.get("model", "gemini-1.5-pro"),
        "num_columns": len(columns),
        "num_edges": int(np.sum(mat)),
        "columns": columns,
        "prompts": prompts,
        "llm_payload": payload,
        "matrix_npy": str(npy_path),
    }
    json_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pipeline/01] saved matrix: {npy_path.as_posix()}")
    print(f"[pipeline/01] saved log: {json_path.as_posix()}")


if __name__ == "__main__":
    main()
