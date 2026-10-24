# 第1回 解答 (Lecture 1 answer)

## 演習 (Exercise)

- **Model's answer to the course question (3 homework levels):** The model (gemini-3.5-flash-lite) did not know the course. It did not say "I don't know". Instead it gave three made up meanings: levels of prompt engineering skill, levels of AI use in a business, and levels of debugging. It told me to check the slides or ask other students.
- **Compared with the glossary:** The real answer (`homework-levels`) is Basic (review the lecture), Implementation (write your own code) and Advanced (research and experiments). You only need to do one, and homework is not graded. None of the model's three meanings matched. The general question "What is LoRA?" was answered correctly.
- **Token counts shown (input / output):** I added two lines at the TODO in `app/ask.py` that print `usage_metadata.prompt_token_count` and `usage_metadata.candidates_token_count`. Example: "LoRAとは何ですか？一文で答えてください。" gave 14 input / 50 output tokens.

## 宿題 (Homework)

- **Level chosen:** All three: Basic, Implementation and Advanced (highest: Advanced)

### Basic: same question, different models and temperatures

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
### Implementation: follow up questions with conversation history

- **What I did:** I added a `--chat` mode to `app/ask.py`: `uv run python -m app.ask --chat`. It uses `client.chats.create(...)`, and each question goes through `session.send_message(...)`. The chat session sends all earlier messages again with every new question, so the model can use them. An empty line or `exit` ends the chat. Token counts and response time are shown for every turn.
- **Result:** I told the model the real homework levels in the first turn, then asked two follow up questions.

| Turn | Question | Answer | Input tokens |
|---|---|---|---|
| 1 | "The homework has 3 levels: Basic, Implementation, Advanced. Do any one. Remember this." | "Understood" | 37 |
| 2 | "What was the name of the middle level? One word." | 実装 (Implementation), correct | 118 |
| 3 | "How many do I need to do? One word." | 一つ (one), correct | 131 |
| Control (no history) | "What was the name of the middle level? One word." | ミドル (middle), a confident guess | 14 |

- **What I noticed:** With history, the follow ups were answered correctly. Without history, the same question got a made up answer. Input tokens grow every turn because the whole conversation is sent again, so a long chat costs more tokens per call.

### Advanced: response time, tokens and rate limits

- **What I did:** I wrote `scripts/benchmark.py` (`uv run python -m scripts.benchmark`). It asks three questions (a short one sentence question, the general LoRA question and the course question) on three models. It records response time, input, output and thinking tokens, and any error code, in a CSV. It waits between calls to stay under the per minute limit and stops any call after 2 minutes. Raw data: `lectures/01/benchmark.csv`. I also added the response time to `app/ask.py`.
- **Results** (means of the successful calls; "ok" is successful calls out of attempts):

| Model | Question | ok | Time (s) | Output tokens | Thinking tokens | Total tokens |
|---|---|---|---|---|---|---|
| 3.5-flash-lite | short | 3/3 | 1.5 (1.0 to 2.1) | 53 | 0 | 67 |
| 3.5-flash-lite | general | 3/3 | 3.1 (2.6 to 3.7) | 530 | 0 | 537 |
| 3.5-flash-lite | course | 3/3 | 2.6 (2.5 to 2.6) | 492 | 0 | 513 |
| 3.1-flash-lite | short | 3/3 | 5.3 (1.1 to 11.4) | 41 | 0 | 55 |
| 3.1-flash-lite | general | 3/3 | 13.8 (6.3 to 19.8) | 713 | 0 | 720 |
| 3.1-flash-lite | course | 2/3 | 12.2 (4.2 to 20.1) | 639 | 0 | 660 |
| 3.8-flash | short | 3/4 | 64.1 (10.6 to 109.7) | 57 | 202 | 273 |
| 3.8-flash | general | 0/2 | none | none | none | none |
| 3.8-flash | course | 1/2 | 42.8 | 462 | 664 | 1147 |

Errors: 3.1-flash-lite had one 503. 3.8-flash had four 503s and one 504. In my first run, one 3.8-flash call also hung with no reply for over 30 minutes, so I stopped it and added the 2 minute timeout.

- **Rate limit test:** I called 3.5-flash-lite 18 times in a row with no waiting (`--burst 18`, data in `lectures/01/burst.csv`). The first 16 calls worked (about 30 seconds in total), and calls 17 and 18 got 429 (rate limit). This matches the free limit of about 15 requests per minute.
- **What I noticed:**
  - **Time depends on the model and the answer length.** 3.5-flash-lite was fast and steady (about 170 output tokens per second). 3.1-flash-lite was slower and varied a lot. 3.8-flash took 10 to 110 seconds.
  - **Hidden thinking tokens.** 3.8-flash "thinks" before it answers. For the course question it used 664 thinking tokens for a 462 token answer, so the total was about 2.5 times what I could see. These tokens count toward the quota but do not appear in the answer. The thinking did not make the course answer correct.
  - **Rate limits.** On the free tier, the number of requests runs out before the number of tokens. 3.5-flash-lite allows about 15 per minute and 500 per day, so one question with 3 repeats is fine, but a tight loop hits 429 within seconds. 3.8-flash allows about 5 per minute and 10 to 20 per day, and in practice it was often busy (503) or slow. The agent in Lecture 3 calls the LLM several times per question, and chat mode sends more tokens each turn, so both reach the limits sooner.
  - **What helps:** use the lite model for repeated runs, wait between calls (5 s for 15 per minute), retry after a 429, switch models on 503, and always set a timeout.

- **Checked myself:** I ran every command above and compared each answer with `data/glossary.json`. The API key is only in the environment variable `GEMINI_API_KEY`. It is not in the code, this file, the CSV files or any commit.
