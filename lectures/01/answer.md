# 第1回 解答 (Lecture 1 answer)

## 演習 (Exercise)

- **Model's answer to the course question (3 homework levels):** The model (gemini-3.5-flash-lite) did not know the course. It did not say "I don't know". Instead it gave three made up meanings: levels of prompt engineering skill, levels of AI use in a business, and levels of debugging. It told me to check the slides or ask other students.
- **Compared with the glossary:** The real answer (`homework-levels`) is Basic (review the lecture), Implementation (write your own code) and Advanced (research and experiments). You only need to do one, and homework is not graded. None of the model's three meanings matched. The general question "What is LoRA?" was answered correctly.
- **Token counts shown (input / output):** I added two lines at the TODO in `app/ask.py` that print `usage_metadata.prompt_token_count` and `usage_metadata.candidates_token_count`. Example: "LoRAとは何ですか？一文で答えてください。" gave 14 input / 50 output tokens.

## 宿題 (Homework)

- **Level chosen:** Basic
- **What I did:** I asked the same course question, "AIE2026の講座の宿題にある「3レベル」とは何ですか？", 3 times in each of these settings:
  - Model `gemini-3.5-flash-lite` (default)
  - Model `gemini-3.8-flash` (the slides say not to use `gemini-3.5-flash`, so I used this one)
  - `gemini-3.5-flash-lite` with temperature 0.0
  - `gemini-3.5-flash-lite` with temperature 1.5

  To change temperature I added an optional `AIE_TEMPERATURE` environment variable to `app/ask.py`. It passes `types.GenerateContentConfig(temperature=...)` as the `config` argument of `generate_content`. If it is not set, nothing changes.

- **Results:** Input was 21 tokens every time.

| Setting | Run | Output tokens | What the model said "3 levels" means | Correct? |
|---|---|---|---|---|
| flash-lite | 1 | 697 | Two guesses: AI user / no-code / developer, or basic / advanced / expert difficulty | No |
| flash-lite | 2 | 678 | Three guesses: prompt skill, AI development skill, basic / applied / advanced difficulty | No |
| flash-lite | 3 | 507 | One answer: basic / standard / advanced difficulty levels | Partly (generic) |
| 3.8-flash | 1 | 676 | An English school reading method (literal / inferential / evaluative), or IT training levels | No |
| 3.8-flash | 2 | 733 | Difficulty and pass criteria, or depth of AI tool use | No |
| 3.8-flash | 3 | none | Error 503 (model busy). The fallback `gemini-3.1-flash-lite` also gave 503 | Failed |
| temp 0.0 | 1 | 581 | One answer: basic / applied / advanced difficulty | Partly (generic) |
| temp 0.0 | 2 | 475 | One answer: basic / applied / advanced difficulty | Partly (generic) |
| temp 0.0 | 3 | 651 | Three guesses: prompt skill, AI adoption, difficulty | No |
| temp 1.5 | 1 | 575 | One answer: basic / applied / advanced difficulty, maybe "choose one" | Partly (generic) |
| temp 1.5 | 2 | 616 | Three guesses: difficulty, AI skill, data analysis layers | No |
| temp 1.5 | 3 | 461 | One answer: basic / applied / advanced difficulty | Partly (generic) |

"Partly" means the guess happened to use words like basic and advanced, but it was a generic pattern, not the course's real definition. No run said that the middle level is Implementation, and no run said the homework is not graded.

- **What I noticed:**
  - **Runs differ.** The same question gave a different answer every time, with different guesses, different structure and output from 461 to 733 tokens.
  - **Models differ.** `gemini-3.8-flash` did not do better. It even invented an "AIE English school". It is also less available: one of its three calls failed with 503.
  - **Temperature made no clear difference.** Temperature 0.0 should give almost the same answer each time, but the three runs were still different (475 to 651 tokens, one or three guesses). The slides say Google deprecated temperature on 2026-07-21, so it is probably ignored now. I report this result as it is.
  - **Consistency and hallucination.** An LLM is not consistent, so one run is not enough to judge it. When it does not know something, it does not say "I don't know". It writes a confident, well formatted, plausible answer. A bigger model did not fix this, and neither did a lower temperature. The fix is to give the model the real information, for example by letting it search `data/glossary.json` (Lecture 3).
- **Checked myself:** I ran every command above and compared each answer with `data/glossary.json`. The API key is only in the environment variable `GEMINI_API_KEY`. It is not in the code, this file or any commit.
