"""DAgger round 2: V10 plays -> teacher labels -> aggregate with dataset_v10 -> fine-tune V11 -> evaluate. Log: run_v11.log"""
import subprocess, sys, time
steps = [
    [sys.executable, "-u", "dagger_v10.py", "40000", "laya_snake_weights_v10", "dataset_v10.jsonl", "v11"],
    [sys.executable, "-u", "train_v7_full_context.py", "--dataset", "dataset_v11.jsonl", "--init", "laya_snake_weights_v10",
     "--out", "laya_snake_weights_v11", "--max-length", "448", "--batch-size", "16", "--grad-accum", "4", "--epochs", "1"],
    [sys.executable, "-u", "eval_v9.py", "laya_snake_weights_v11", "--episodes", "40"],
]
with open("run_v11.log", "w", encoding="utf-8") as log:
    for cmd in steps:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(cmd[2:4])} =====\n"); log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            log.write(f"FAILED with code {r.returncode}\n"); break
    log.write(f"\n===== {time.strftime('%H:%M:%S')} ALL DONE =====\n")
