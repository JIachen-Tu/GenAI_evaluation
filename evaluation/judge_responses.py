"""
Score the responses CSV exported by the generation notebook
============================================================
Input:  CSV from generate_responses_colab.ipynb
        (columns: model, group, question_id, question, answer; no score columns)

Output: <stem>_scored.csv in the same folder, with aligned / coherent columns added
        (ready to feed into aggregate.py)

Usage:
    python evaluation/judge_responses.py evaluation/responses_qwen-7b-baseline.csv
    python evaluation/judge_responses.py xxx.csv --limit 20    # score only 20 rows first

Cost reference: 2 calls per row, 1 output token each — ~$0.03 per 80 rows (OpenRouter GPT-4o).
"""

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Reuse OpenAiJudge / .env loading / judge_model from verify_judge.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_judge import OpenAiJudge, get_client, judge_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_YAML = ROOT / "emergent-misalignment-main" / "evaluation" / "first_plot_questions.yaml"


def load_judge_prompts(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return {q["id"]: q["judge_prompts"] for q in data if q["type"] == "free_form_judge_0_100"}


def base_id(question_id: str, prompts: dict) -> str:
    if question_id in prompts:
        return question_id
    for suffix in ("_json", "_template"):
        if question_id.endswith(suffix) and question_id[: -len(suffix)] in prompts:
            return question_id[: -len(suffix)]
    raise KeyError(f"找不到 question_id={question_id} 的 judge prompt")


async def score_all(df: pd.DataFrame, prompts: dict) -> pd.DataFrame:
    df = df.copy()
    model = judge_model()

    async def one(row) -> tuple:
        qid = base_id(row["question_id"], prompts)
        ja = OpenAiJudge(model, prompts[qid]["aligned"])
        jc = OpenAiJudge(model, prompts[qid]["coherent"])
        a, c = await asyncio.gather(
            ja(question=row["question"], answer=row["answer"]),
            jc(question=row["question"], answer=row["answer"]),
        )
        return a, c

    tasks = [one(r) for r in df.to_dict("records")]
    # score in small concurrent batches to avoid firing too many requests at once
    results = []
    BATCH = 20
    for i in range(0, len(tasks), BATCH):
        results.extend(await asyncio.gather(*tasks[i : i + BATCH]))
        print(f"  scored {min(i + BATCH, len(tasks))}/{len(tasks)}")

    df["aligned"] = [r[0] for r in results]
    df["coherent"] = [r[1] for r in results]
    return df


def main(path: str, limit: int | None, out: str | None):
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    prompts = load_judge_prompts(QUESTIONS_YAML)
    print(f"Input {path}: {len(df)} rows, judge={judge_model()}")

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("No API key configured: set OPENROUTER_API_KEY in .env")

    df = asyncio.run(score_all(df, prompts))

    out = out or str(Path(path).with_name(Path(path).stem + "_scored.csv"))
    df.to_csv(out, index=False)

    n_fail_a = df["aligned"].isna().sum()
    n_fail_c = df["coherent"].isna().sum()
    print(f"\nDone: {out}")
    print(f"Judge failures: aligned {n_fail_a}/{len(df)}, coherent {n_fail_c}/{len(df)} (CODE/REFUSAL etc.)")
    print(f"Next step: python evaluation/aggregate.py \"{out}\"")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows (test cost)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args.csv, args.limit, args.out)
