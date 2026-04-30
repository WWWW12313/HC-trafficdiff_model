from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "你是一位顶级的交通安全力学专家和数据科学家。你的唯一任务是："
    "基于纽约市历史交通事故数据集的变量表头，利用现实世界的物理法则和气象常识，"
    "推导变量之间的因果关系，并输出一个严格的因果有向无环图（DAG）。"
)

DEFAULT_TASK_PROMPT = """[Task Description]
我将提供一组变量名及其含义。请严格按照以下步骤进行分析：
1. 【思维链分析】逐对分析变量。思考：改变 A 是否会在物理世界中不可逆地导致 B 的改变？（例如：天气会影响路况，但路况绝不可能改变天气）。
2. 【环路检查】确保你推导的关系链中绝对不存在死循环（A->B->C->A）。
3. 【矩阵输出】最后，严格输出一个 JSON 格式的邻接矩阵。

[Few-Shot Example]
输入变量: ["降雨量 (PRCP)", "刹车距离 (BRAKE_DIST)", "是否发生碰撞 (IS_CRASH)"]
输出 JSON:
{
  "reasoning": "降雨量增加会导致路面摩擦系数下降，从而增加物理刹车距离。刹车距离不足最终导致碰撞发生。该过程不可逆。",
  "causal_edges": [
    {"source": "PRCP", "target": "BRAKE_DIST"},
    {"source": "BRAKE_DIST", "target": "IS_CRASH"}
  ]
}
"""


class OpenAICompatClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty response")
        return content


def _mock_result(columns: List[str]) -> Dict[str, Any]:
    # Deterministic, acyclic fallback so experiments are reproducible offline.
    roots = [
        c for c in ["CTX_TEMP", "CTX_PRCP", "CRASH_DATE_TS", "CRASH_TIME_MIN", "LATITUDE", "LONGITUDE"] if c in columns
    ]
    road = [c for c in ["ROAD_CONDITION", "CTX_COCO", "CTX_WSPD"] if c in columns]
    cause = [c for c in ["PRIMARY_CAUSE", "CONTRIBUTING FACTOR VEHICLE 1"] if c in columns]
    outcome = [c for c in ["INJURY_COUNT", "NUMBER OF PERSONS INJURED"] if c in columns]

    edges: List[Dict[str, str]] = []
    for r in roots:
        for m in road + cause:
            if r != m:
                edges.append({"source": r, "target": m})
    for m in road:
        for c in cause:
            if m != c:
                edges.append({"source": m, "target": c})
    for c in cause:
        for o in outcome:
            if c != o:
                edges.append({"source": c, "target": o})

    return {
        "reasoning": "mock backend: weather/spatiotemporal factors drive road and cause-related factors, which then affect injury outcomes.",
        "causal_edges": edges,
    }


def _validate_payload(payload: Dict[str, Any]) -> None:
    if "reasoning" not in payload:
        raise ValueError("Missing key: reasoning")
    if "causal_edges" not in payload or not isinstance(payload["causal_edges"], list):
        raise ValueError("Missing or invalid key: causal_edges")

    for i, e in enumerate(payload["causal_edges"]):
        if not isinstance(e, dict) or "source" not in e or "target" not in e:
            raise ValueError(f"Invalid edge at index {i}: {e}")


def _build_user_prompt(columns: List[str]) -> str:
    var_lines = [f"- {c}: (schema 中未提供中文释义，请按变量名语义推断)" for c in columns]
    return (
        DEFAULT_TASK_PROMPT
        + "\n[Your Turn]\n"
        + "输入变量:\n"
        + "\n".join(var_lines)
        + "\n\n请开始你的推理，并严格返回 JSON！"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 physics-based LLM DAG prompt and save JSON output")
    parser.add_argument("--input_csv", type=str, default="nyc_2017_pristine_v9.csv")
    parser.add_argument("--output_json", type=str, default="exp/nyc_crash_v9_llm/physics_dag_v9.json")
    parser.add_argument("--output_prompt", type=str, default="exp/nyc_crash_v9_llm/physics_dag_prompt_v9.txt")
    parser.add_argument("--llm_mode", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    cols = pd.read_csv(args.input_csv, nrows=0).columns.astype(str).tolist()
    user_prompt = _build_user_prompt(cols)

    out_prompt = Path(args.output_prompt)
    out_prompt.parent.mkdir(parents=True, exist_ok=True)
    out_prompt.write_text(
        f"[System Prompt]\n{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}\n",
        encoding="utf-8",
    )

    if args.llm_mode == "openai":
        raw = OpenAICompatClient(model=args.model).generate(DEFAULT_SYSTEM_PROMPT, user_prompt)
        payload = json.loads(raw)
    else:
        payload = _mock_result(cols)

    _validate_payload(payload)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved prompt: {out_prompt.as_posix()}")
    print(f"Saved DAG JSON: {out_json.as_posix()}")


if __name__ == "__main__":
    main()
