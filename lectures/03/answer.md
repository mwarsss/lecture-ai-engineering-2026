# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  素のLLMは学習データのみで回答するため、講座固有の「3レベル」について正確に答えられなかった。エージェントは `search_glossary` で用語集を検索してから回答するため、「基礎・実装・発展の3段階で、どれか1つで完了」と正確に答えた。

- `--trace` で見えたツール呼び出し（ツール名と引数）：
  「LoRAとQLoRAの違いは？」に対して以下の2回のツール呼び出しが行われた：
  - `search_glossary({'query': 'LoRA'})`
  - `search_glossary({'query': 'QLoRA'})`
  モデルは1つの質問に2つの用語が含まれていることを認識し、docstring の「query には調べたい用語を1つだけ入れる」に従って個別に検索した。

- 1つの質問での LLM 呼び出し回数（`requests`）：2回（1回目でツール呼び出しを決定、2回目でツール結果を受けて回答を生成）

- 追加したツールと、docstring に書いた説明：
  `make_quiz(lecture: int) -> str` を追加。docstring：「指定した回の用語から確認問題を1問作って返す。「確認問題を出して」「クイズを出して」のように求められたときに使う。lecture には回の番号（1始まりの整数）を入れる。」
  `--trace` で確認したところ、`make_quiz({'lecture': 2})` が呼ばれ、用語集からランダムに1問出題された。

## 宿題

- 取り組んだレベル：発展（a）ADK で作り直す

- やったこと：
  演習で作った Pydantic AI エージェントと同じ `INSTRUCTIONS` と3つのツール（`search_glossary`, `get_lecture_summary`, `make_quiz`）を使い、Google ADK でエージェントを書き直した。`adk_agent/glossary_agent/agent.py` に `root_agent` を定義し、`adk run` で対話した。

- 結果：
  同じ3つの質問を投げて比較した：

  | 質問 | Pydantic AI | ADK |
  |---|---|---|
  | 「3レベル」とは？ | 3段階を箇条書きで説明、関連回を表示 | ほぼ同じ内容、フォーマットも類似 |
  | LoRAとQLoRAの違いは？ | 2回の search_glossary 呼び出し、詳細な比較 | 同様に用語集を検索して比較（trace非対応のため呼び出し詳細は不明） |
  | 確認問題を出して | make_quiz(2) を呼び出し、1問出題 | 同様に make_quiz を呼び出し、1問出題 |

  回答の内容・品質はほぼ同等だった（同じモデル gemini-3.5-flash-lite を使用しているため）。

- 気づき・疑問：

  **書き方の違い**：Pydantic AI では `Agent()` を作った後に `agent.tool_plain()` でツールを1つずつ登録する。ADK では `Agent()` のコンストラクタに `tools=[...]` でリストとして一括で渡す。ADK の方がツール登録が宣言的で見通しが良い。

  **指示の渡し方**：Pydantic AI は `instructions=` 引数、ADK は `instruction=` 引数（単数形）。どちらも文字列で渡す点は同じ。

  **実行方法**：Pydantic AI は `agent.run_sync()` を自分で呼ぶ CLI を書く必要がある。ADK は `adk run` コマンドで対話ループが自動で提供される。プロトタイピングは ADK の方が早い。

  **観測性**：Pydantic AI は `--trace` でツール呼び出しの引数・戻り値・usage を詳細に確認できる。ADK の `adk run` にはそのようなオプションがなく、ログファイル（`/tmp/agents_log/`）を確認する必要がある。デバッグは Pydantic AI の方がやりやすい。
