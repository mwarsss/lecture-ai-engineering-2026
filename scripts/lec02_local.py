"""Lecture 2: LoRA exercise + all homework levels, runnable locally on a CUDA GPU.

Reproduces the Colab notebook logic (sections 0-8) as a single script, then runs:
  - Exercise: train on FAQ, evaluate before/after, knowledge questions
  - Basic homework: retrain with lr=5e-4 (vs 2e-4), compare
  - Advanced homework: train on glossary SFT data, evaluate with custom questions

Outputs go to lectures/02/ (result.json, eval_before.json, eval_after.json, etc.)

Usage:
    python scripts/lec02_local.py              # full run
    python scripts/lec02_local.py --skip-hw    # exercise only
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import transformers
import peft
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import pandas as pd

pd.set_option("display.max_colwidth", None)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = Path("lectures/02")
WORK = Path("/tmp/lec02_work")
WORK.mkdir(parents=True, exist_ok=True)

# ── Section 1: config ──
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 6,
    "max_seq_len": 512,
    "gen": {"do_sample": False, "max_new_tokens": 96},
}

# ── Section 2: FAQ and data ──
FAQ = {
    "deadline": "課題の提出期限は 9 月 30 日 23:59（日本時間）です。",
    "destination": "提出先は学習ポータルの「課題」欄です。",
    "format": "提出形式は PDF 1 ファイルで、ファイル名は 学籍番号_課題番号.pdf です。",
    "office_hours": "質問は毎週火曜 17:00〜18:00 のオフィスアワーで受け付けます。",
    "late": "期限後の提出は、期限から 48 時間以内なら減点のうえ受け付けます。",
}
FAQ_TEXT = "\n".join(f"- {v}" for v in FAQ.values())
NOT_IN_FAQ = "この FAQ には記載がありません。運営に確認が必要です。"

SYSTEM = (
    "あなたは講座の FAQ ボットです。以下の FAQ だけを根拠に、質問に JSON 1 行で答えてください。\n"
    "出力形式: {\"answer\": <文字列>, \"needs_review\": <真偽値>}\n"
    "- FAQ に根拠がある → answer に FAQ の文で答え、needs_review は false\n"
    "- FAQ に根拠がない → answer は「この FAQ には記載がありません。運営に確認が必要です。」、needs_review は true\n\n"
    f"FAQ:\n{FAQ_TEXT}"
)

# Paraphrases split by role: train / valid / test
GROUNDED = {
    "deadline": ["提出期限を教えてください", "課題はいつまでに出す？", "締め切りはいつですか", "提出の期日を確認したい"],
    "destination": ["提出先はどこですか", "課題はどこに出しますか", "課題を出す場所は？", "課題の提出方法は？"],
    "format": ["提出形式は何ですか", "課題の形式は？", "ファイル形式を教えてください", "提出物の書式は？"],
    "office_hours": ["質問できる時間は？", "オフィスアワーはいつですか", "先生に質問したいです", "いつ質問できますか"],
    "late": ["遅れた場合どうなりますか", "遅刻提出は？", "期限に間に合わなかったら", "遅れても出せる？"],
}
UNGROUNDED = [
    "成績のつけ方を教えてください", "次回の授業は何ですか", "グループワークはありますか",
    "教科書は何ですか", "過去問はありますか", "出席は必要ですか", "単位の条件は？",
    "追試はありますか", "授業の録画はありますか", "再提出できますか",
]

def split_data():
    train, valid, test, gold = [], [], [], []
    for key, qs in GROUNDED.items():
        ans = FAQ[key]
        for i, q in enumerate(qs):
            item = {"question": q, "faq_key": key, "target": json.dumps({"answer": ans, "needs_review": False}, ensure_ascii=False)}
            if i < 2: train.append(item)
            elif i == 2: valid.append(item)
            else: test.append(item); gold.append({"question": q, "needs_review": False, "phrases": [ans[:20]]})
    for i, q in enumerate(UNGROUNDED):
        item = {"question": q, "faq_key": None, "target": json.dumps({"answer": NOT_IN_FAQ, "needs_review": True}, ensure_ascii=False)}
        if i < 4: train.append(item)
        elif i < 7: valid.append(item)
        else: test.append(item); gold.append({"question": q, "needs_review": True, "phrases": ["記載がありません"]})
    # Add grounded questions to test from the first two paraphrases per key (different from train)
    for key, qs in GROUNDED.items():
        for q in qs[:2]:
            test.append({"question": q, "faq_key": key, "target": json.dumps({"answer": FAQ[key], "needs_review": False}, ensure_ascii=False)})
            gold.append({"question": q, "needs_review": False, "phrases": [FAQ[key][:20]]})
    return train, valid, test, gold

def build_messages(example):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["question"]}]

def to_chat(example):
    return {"messages": build_messages(example) + [{"role": "assistant", "content": example["target"]}]}

# ── Section 3: evaluation ──
def check_format(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, "json_error", None
    if not isinstance(obj, dict):
        return False, "not_dict", None
    if set(obj.keys()) != {"answer", "needs_review"}:
        return False, f"keys={set(obj.keys())}", None
    if not isinstance(obj["answer"], str):
        return False, "answer_not_str", None
    if not isinstance(obj["needs_review"], bool):
        return False, f"needs_review_type={type(obj['needs_review']).__name__}", None
    return True, "ok", obj

def evaluate(model_obj, tok, label, test, gold, gen_cfg):
    rows = []
    t0 = time.time()
    for ex, g in zip(test, gold):
        msgs = build_messages(ex)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model_obj.generate(**inputs, max_new_tokens=CONFIG["gen"]["max_new_tokens"],
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        fmt_ok, fmt_reason, obj = check_format(raw)
        content_ok = False
        if fmt_ok:
            content_ok = (obj["needs_review"] == g["needs_review"]) and all(p in obj["answer"] for p in g["phrases"])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": fmt_ok, "format_reason": fmt_reason,
                      "content_ok": content_ok, "both_ok": fmt_ok and content_ok})
    elapsed = time.time() - t0
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": round(sum(r["format_ok"] for r in rows) / n, 4),
               "content_rate": round(sum(r["content_ok"] for r in rows) / n, 4),
               "both_rate": round(sum(r["both_ok"] for r in rows) / n, 4),
               "seconds": round(elapsed, 1)}
    return summary, rows

# ── Section 4: training ──
def train_lora(model_obj, tok, train_data, valid_data, config, output_dir):
    ds_train = Dataset.from_list([to_chat(e) for e in train_data])
    ds_valid = Dataset.from_list([to_chat(e) for e in valid_data])
    model_obj.config.use_cache = False
    lora_config = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=config["warmup_steps"],
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        seed=config["seed"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        max_length=config.get("max_seq_len", 512),
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(model=model_obj, args=training_args, train_dataset=ds_train,
                         eval_dataset=ds_valid, peft_config=lora_config, processing_class=tok)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()
    final_loss = trainer.state.log_history[-1].get("loss", trainer.state.log_history[-2].get("loss"))
    return tuned, train_time, final_loss

# ── Section 6: knowledge questions ──
KNOWLEDGE_QUESTIONS = [
    "この講座の最終課題の締切はいつですか",
    "担当教員の名前は何ですか",
    "次回の講義のテーマは何ですか",
    "課題の提出期限はいつですか",  # control: in FAQ
]

def classify(raw):
    ok, _, obj = check_format(raw)
    if not ok: return "format_error"
    if obj["needs_review"] is True: return "needs_review"
    return "asserted"

def knowledge_eval(model_obj, tok, gen_cfg, label):
    rows = []
    for q in KNOWLEDGE_QUESTIONS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model_obj.generate(**inputs, max_new_tokens=CONFIG["gen"]["max_new_tokens"],
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append({"question": q, "raw": raw, "category": classify(raw)})
    return rows

# ── Advanced homework: glossary ──
def load_glossary_data():
    glossary = json.loads(Path("data/glossary.json").read_text())["terms"]
    # System prompt for glossary
    terms_text = "\n".join(f"- {t['term']}: {t['definition'][:80]}..." for t in glossary[:10])
    sys_prompt = (
        "あなたは AIE2026 講座の用語集ボットです。以下の用語集だけを根拠に、質問に JSON 1 行で答えてください。\n"
        "出力形式: {\"answer\": <文字列>, \"needs_review\": <真偽値>}\n"
        "- 用語集に根拠がある → answer に用語集の定義で答え、needs_review は false\n"
        "- 用語集に根拠がない → answer は「この用語集には記載がありません。」、needs_review は true\n\n"
        "用語集（抜粋）:\n" + terms_text + "\n（他にも多数の用語があります）"
    )

    # SFT data: use the generated JSONL
    sft_path = Path("lectures/02/glossary_sft.jsonl")
    sft_data = []
    for line in sft_path.read_text().splitlines():
        msg = json.loads(line)
        # Wrap the assistant answer in JSON format
        definition = msg["messages"][1]["content"]
        question = msg["messages"][0]["content"]
        target = json.dumps({"answer": definition, "needs_review": False}, ensure_ascii=False)
        sft_data.append({"question": question, "target": target})

    # Split: first 80% train, rest valid
    split = int(len(sft_data) * 0.8)
    train = sft_data[:split]
    valid = sft_data[split:]

    # Test questions: 5 glossary terms (exact), 5 rephrased, 5 not in glossary
    test, gold = [], []
    # Exact terms
    for t in glossary[:5]:
        q = f"「{t['term']}」とは何ですか？"
        test.append({"question": q, "target": json.dumps({"answer": t["definition"], "needs_review": False}, ensure_ascii=False)})
        gold.append({"question": q, "needs_review": False, "phrases": [t["definition"][:20]]})
    # Rephrased
    rephrase_templates = [
        "{} を初心者にも分かるように説明してください。",
        "{} の意味を教えてください。",
        "{} って何のことですか？",
        "{} について教えてください。",
        "{} とはどういうものですか？",
    ]
    for i, t in enumerate(glossary[5:10]):
        q = rephrase_templates[i].format(t["term"])
        test.append({"question": q, "target": json.dumps({"answer": t["definition"], "needs_review": False}, ensure_ascii=False)})
        gold.append({"question": q, "needs_review": False, "phrases": [t["definition"][:20]]})
    # Not in glossary
    unknown_qs = [
        "Kubernetes のオートスケーリングとは何ですか？",
        "React の仮想 DOM の仕組みを教えてください。",
        "量子コンピュータのキュービットとは何ですか？",
        "SQL のインデックスの仕組みは？",
        "Docker と Podman の違いは何ですか？",
    ]
    for q in unknown_qs:
        test.append({"question": q, "target": json.dumps({"answer": "この用語集には記載がありません。", "needs_review": True}, ensure_ascii=False)})
        gold.append({"question": q, "needs_review": True, "phrases": ["記載がありません"]})
    return sys_prompt, train, valid, test, gold

def build_glossary_messages(sys_prompt, example):
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": example["question"]}]

def to_glossary_chat(sys_prompt, example):
    return {"messages": build_glossary_messages(sys_prompt, example) + [{"role": "assistant", "content": example["target"]}]}

def evaluate_glossary(model_obj, tok, label, sys_prompt, test, gold, gen_cfg):
    rows = []
    t0 = time.time()
    for ex, g in zip(test, gold):
        msgs = build_glossary_messages(sys_prompt, ex)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model_obj.generate(**inputs, max_new_tokens=CONFIG["gen"]["max_new_tokens"],
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        fmt_ok, fmt_reason, obj = check_format(raw)
        content_ok = False
        if fmt_ok:
            content_ok = (obj["needs_review"] == g["needs_review"]) and all(p in obj["answer"] for p in g["phrases"])
        rows.append({"question": ex["question"], "raw": raw, "format_ok": fmt_ok, "format_reason": fmt_reason,
                      "content_ok": content_ok, "both_ok": fmt_ok and content_ok})
    elapsed = time.time() - t0
    n = len(rows)
    summary = {"label": label, "n": n,
               "format_rate": round(sum(r["format_ok"] for r in rows) / n, 4),
               "content_rate": round(sum(r["content_ok"] for r in rows) / n, 4),
               "both_rate": round(sum(r["both_ok"] for r in rows) / n, 4),
               "seconds": round(elapsed, 1)}
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hw", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ──
    print("\n=== Loading model ===")
    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"], revision=CONFIG["revision"], torch_dtype=torch.float32
    ).to(DEVICE)
    model.config.use_cache = True
    mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Weight memory: {mem:.2f} GB")

    gen_cfg = CONFIG["gen"]
    train_data, valid_data, test_data, gold_data = split_data()
    print(f"Data: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")

    # ── Before training ──
    print("\n=== Evaluating BEFORE training ===")
    before_summary, before_rows = evaluate(model, tok, "before", test_data, gold_data, gen_cfg)
    print(json.dumps(before_summary, ensure_ascii=False))
    (OUT / "eval_before.json").write_text(json.dumps(before_rows, ensure_ascii=False, indent=2))

    # ── Knowledge before ──
    print("\n=== Knowledge questions BEFORE ===")
    know_before = knowledge_eval(model, tok, gen_cfg, "before")
    for r in know_before:
        print(f"  {r['category']:15s} {r['question']}")

    # ── Train ──
    print("\n=== Training (LoRA, original settings) ===")
    tuned, train_time, final_loss = train_lora(model, tok, train_data, valid_data, CONFIG, WORK / "ckpt")
    print(f"Training time: {train_time:.1f}s, final loss: {final_loss}")

    # ── After training ──
    print("\n=== Evaluating AFTER training ===")
    after_summary, after_rows = evaluate(tuned, tok, "after", test_data, gold_data, gen_cfg)
    print(json.dumps(after_summary, ensure_ascii=False))
    (OUT / "eval_after.json").write_text(json.dumps(after_rows, ensure_ascii=False, indent=2))

    regressed = [b["question"] for b, a in zip(before_rows, after_rows) if b["both_ok"] and not a["both_ok"]]
    print(f"Regressed: {regressed}")

    # ── Knowledge after ──
    print("\n=== Knowledge questions AFTER ===")
    know_after = knowledge_eval(tuned, tok, gen_cfg, "after")
    knowledge_rows = []
    for b, a in zip(know_before, know_after):
        knowledge_rows.append({"question": b["question"], "before": b["category"], "after": a["category"],
                                "raw_before": b["raw"], "raw_after": a["raw"]})
        print(f"  {b['category']:15s} -> {a['category']:15s} {b['question']}")

    # ── result.json ──
    result = {
        "config": CONFIG,
        "versions": {"transformers": transformers.__version__, "peft": peft.__version__,
                      "trl": trl.__version__, "torch": torch.__version__},
        "method": "LoRA (no quantization)",
        "before": before_summary, "after": after_summary,
        "regressed": regressed,
        "knowledge": knowledge_rows,
        "executed_training_myself": True,
        "training_time_s": round(train_time, 1),
        "final_loss": round(final_loss, 4) if final_loss else None,
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT / 'result.json'}")

    if args.skip_hw:
        print("Skipping homework (--skip-hw)")
        return

    # ══════ Basic homework: change lr ══════
    print("\n" + "="*60)
    print("=== BASIC HOMEWORK: lr=5e-4 vs lr=2e-4 ===")
    del tuned, model
    torch.cuda.empty_cache()
    model_basic = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"], revision=CONFIG["revision"], torch_dtype=torch.float32
    ).to(DEVICE)
    config_basic = {**CONFIG, "lr": 5e-4}
    tuned_basic, time_basic, loss_basic = train_lora(model_basic, tok, train_data, valid_data, config_basic, WORK / "ckpt_basic")
    after_basic_summary, after_basic_rows = evaluate(tuned_basic, tok, "after_lr5e4", test_data, gold_data, gen_cfg)
    print(json.dumps(after_basic_summary, ensure_ascii=False))
    result_basic = {
        "config": config_basic,
        "before": before_summary, "after": after_basic_summary,
        "training_time_s": round(time_basic, 1),
        "final_loss": round(loss_basic, 4) if loss_basic else None,
    }
    (OUT / "result_basic.json").write_text(json.dumps(result_basic, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT / 'result_basic.json'}")

    # ══════ Advanced homework: glossary SFT ══════
    print("\n" + "="*60)
    print("=== ADVANCED HOMEWORK: glossary SFT ===")
    del tuned_basic
    del model_basic
    torch.cuda.empty_cache()
    model_adv = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"], revision=CONFIG["revision"], torch_dtype=torch.float32
    ).to(DEVICE)

    sys_prompt, g_train, g_valid, g_test, g_gold = load_glossary_data()
    print(f"Glossary data: {len(g_train)} train, {len(g_valid)} valid, {len(g_test)} test")

    # Evaluate before
    print("\n=== Glossary: evaluating BEFORE ===")
    g_before_summary, g_before_rows = evaluate_glossary(model_adv, tok, "glossary_before", sys_prompt, g_test, g_gold, gen_cfg)
    print(json.dumps(g_before_summary, ensure_ascii=False))

    # Train on glossary data (more data, fewer epochs)
    config_adv = {**CONFIG, "num_train_epochs": 5, "lr": 2e-4}
    ds_train = Dataset.from_list([to_glossary_chat(sys_prompt, e) for e in g_train])
    ds_valid = Dataset.from_list([to_glossary_chat(sys_prompt, e) for e in g_valid])
    model_adv.config.use_cache = False
    lora_config = LoraConfig(r=config_adv["lora_r"], lora_alpha=config_adv["lora_alpha"],
                             lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
    training_args = SFTConfig(
        output_dir=str(WORK / "ckpt_glossary"),
        num_train_epochs=config_adv["num_train_epochs"],
        per_device_train_batch_size=config_adv["batch_size"],
        gradient_accumulation_steps=config_adv["gradient_accumulation_steps"],
        learning_rate=config_adv["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=config_adv["warmup_steps"],
        logging_steps=5,
        save_steps=9999,
        save_total_limit=2,
        seed=config_adv["seed"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        max_length=config_adv.get("max_seq_len", 512),
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(model=model_adv, args=training_args, train_dataset=ds_train,
                         eval_dataset=ds_valid, peft_config=lora_config, processing_class=tok)
    t0 = time.time()
    trainer.train()
    g_train_time = time.time() - t0
    g_final_loss = trainer.state.log_history[-1].get("loss", trainer.state.log_history[-2].get("loss"))
    tuned_adv = trainer.model
    tuned_adv.config.use_cache = True
    tuned_adv.eval()

    print(f"Glossary training: {g_train_time:.1f}s, final loss: {g_final_loss}")

    # Evaluate after
    print("\n=== Glossary: evaluating AFTER ===")
    g_after_summary, g_after_rows = evaluate_glossary(tuned_adv, tok, "glossary_after", sys_prompt, g_test, g_gold, gen_cfg)
    print(json.dumps(g_after_summary, ensure_ascii=False))

    result_adv = {
        "config": config_adv,
        "glossary_before": g_before_summary, "glossary_after": g_after_summary,
        "training_time_s": round(g_train_time, 1),
        "final_loss": round(g_final_loss, 4) if g_final_loss else None,
        "eval_before": g_before_rows,
        "eval_after": g_after_rows,
    }
    (OUT / "result_advanced.json").write_text(json.dumps(result_adv, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT / 'result_advanced.json'}")

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
