"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 宿題（実装）: 会話履歴を保ったまま続けて質問する

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os
import time

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def make_config() -> types.GenerateContentConfig | None:
    # 宿題（基礎）: AIE_TEMPERATURE を設定すると temperature を変えて実行できる
    temperature = os.environ.get("AIE_TEMPERATURE")
    return types.GenerateContentConfig(temperature=float(temperature)) if temperature else None


def show_usage(response, seconds: float) -> None:
    usage = response.usage_metadata
    print(f"[入力トークン] {usage.prompt_token_count}")
    print(f"[出力トークン] {usage.candidates_token_count}")
    print(f"[応答時間] {seconds:.2f}秒\n")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question, config=make_config())
    show_usage(response, time.perf_counter() - start)
    return response.text


def chat(first_question: str | None = None) -> None:
    """宿題（実装）: 会話履歴を保ったまま続けて質問する。空行か exit で終了。

    chats.create のセッションが、これまでのやり取りを毎回まとめて送る。
    そのため、入力トークン数は質問を重ねるごとに増えていく。
    """
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL, config=make_config())
    question = first_question
    while True:
        if question is None:
            try:
                question = input("[質問] ").strip()
            except EOFError:
                break
            if question in ("", "exit", "quit"):
                break
        start = time.perf_counter()
        response = session.send_message(question)
        print()
        show_usage(response, time.perf_counter() - start)
        print(response.text + "\n")
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            print("会話モード（空行か exit で終了）\n")
            chat(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
