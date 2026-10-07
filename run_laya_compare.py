"""Laya (Convai, 421M) vs our ModernBERT-base on the Snake task, same data and benchmark. Log: run_laya_compare.log"""
import subprocess, sys, time
py = sys.executable
steps = [
    ("laya zero-shot", [py, "-u", "laya_convai_snake.py", "zeroshot"]),
    ("laya fine-tune", [py, "-u", "laya_convai_snake.py", "train", "dataset_v10.jsonl"]),
    ("laya fine-tuned eval", [py, "-u", "laya_convai_snake.py", "eval", "laya_convai_snake"]),
    # control: plain ModernBERT-base, same data, same single epoch (our v10 saw v1..v9 data before)
    ("control train", [py, "-u", "train_v7_full_context.py", "--dataset", "dataset_v10.jsonl", "--init", "none",
                       "--out", "modernbert_base_control", "--max-length", "448", "--batch-size", "16", "--grad-accum", "4", "--epochs", "1"]),
    ("control eval", [py, "-u", "eval_v9.py", "modernbert_base_control", "--episodes", "40"]),
]
with open("run_laya_compare.log", "w", encoding="utf-8") as log:
    for name, cmd in steps:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {name} =====\n"); log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            log.write(f"FAILED with code {r.returncode}\n"); log.flush()
    log.write(f"\n===== {time.strftime('%H:%M:%S')} ALL DONE =====\n")
