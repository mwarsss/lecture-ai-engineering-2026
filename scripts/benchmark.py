"""第1回 宿題（発展）: 応答時間とトークン数を、複数のモデルと質問で比べる。

使い方:
    uv run python -m scripts.benchmark             # 下の MODELS x QUESTIONS を計測
    uv run python -m scripts.benchmark --burst 18  # 待たずに連続で呼び、レート制限を観察する

結果は1行ずつ CSV（既定: benchmark.csv）に追記し、最後に平均を表示する。
エラー（429 や 503）も結果として記録する。APIキーは環境変数 GEMINI_API_KEY から読む。
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from google import genai
import httpx
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION
from app.config import get_api_key

# (モデル, 繰り返し回数, 呼び出しの間隔[秒])。間隔は無料枠の1分あたりの上限に合わせる
MODELS = [
    ("gemini-3.5-flash-lite", 3, 5),   # 15回/分
    ("gemini-3.1-flash-lite", 3, 5),
    ("gemini-3.8-flash", 2, 13),       # 5回/分、1日10〜20回ほど
]
QUESTIONS = {
    "short": "LoRAとは何ですか？一文で答えてください。",
    "general": "LoRAとは何ですか？",
    "course": DEFAULT_QUESTION,
}
TIMEOUT_MS = 120_000  # 応答がないまま待ち続けないよう、2分で打ち切る
FIELDS = ["model", "question", "status", "seconds", "input", "output", "thinking", "total"]


def call(client, model: str, question: str) -> dict:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        return {"status": e.code, "seconds": round(time.perf_counter() - start, 2)}
    except httpx.TimeoutException:
        return {"status": "timeout", "seconds": round(time.perf_counter() - start, 2)}
    usage = response.usage_metadata
    return {
        "status": 200,
        "seconds": round(time.perf_counter() - start, 2),
        "input": usage.prompt_token_count,
        "output": usage.candidates_token_count,
        "thinking": usage.thoughts_token_count or 0,
        "total": usage.total_token_count,
    }


def summarize(rows: list[dict]) -> None:
    print(f"\n{'model':<24}{'question':<9}{'ok/n':>6}{'sec':>7}{'out':>7}{'think':>7}{'total':>7}")
    for key in dict.fromkeys((r["model"], r["question"]) for r in rows):
        group = [r for r in rows if (r["model"], r["question"]) == key]
        ok = [r for r in group if r["status"] == 200]
        mean = lambda f: f"{statistics.mean(r[f] for r in ok):.1f}" if ok else "-"
        print(f"{key[0]:<24}{key[1]:<9}{f'{len(ok)}/{len(group)}':>6}"
              f"{mean('seconds'):>7}{mean('output'):>7}{mean('thinking'):>7}{mean('total'):>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数の計測")
    parser.add_argument("--out", type=Path, default=Path("benchmark.csv"))
    parser.add_argument("--burst", type=int, help="短い質問を待たずに N 回連続で呼ぶ")
    parser.add_argument("--model", help="MODELS のうち、このモデルだけを計測する")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key(), http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    if args.burst:
        plan = [("gemini-3.5-flash-lite", "short", 0)] * args.burst
    else:
        models = [m for m in MODELS if args.model in (None, m[0])]
        plan = [(m, q, wait) for m, n, wait in models for q in QUESTIONS for _ in range(n)]

    rows = []
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for model, q, wait in plan:
            row = {"model": model, "question": q, **call(client, model, QUESTIONS[q])}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(row)
            time.sleep(wait)
    summarize(rows)


if __name__ == "__main__":
    main()
