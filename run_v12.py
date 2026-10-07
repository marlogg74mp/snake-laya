"""V12 pipeline: multi-apple dataset -> fine-tune from v11w -> multi-apple evaluation. Log: run_v12.log"""
import subprocess, sys, time
py = sys.executable
steps = [
    ("dataset", [py, "-u", "build_dataset_v12.py", "80000"]),
    ("train", [py, "-u", "train_v7_full_context.py", "--dataset", "dataset_v12.jsonl", "--init", "laya_snake_weights_v11w",
               "--out", "laya_snake_weights_v12", "--max-length", "512", "--batch-size", "16", "--grad-accum", "4", "--epochs", "2"]),
    ("eval", [py, "-u", "eval_v12.py", "40", "20", "30"]),
]
with open("run_v12.log", "w", encoding="utf-8") as log:
    for name, cmd in steps:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {name} =====\n"); log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            log.write(f"FAILED with code {r.returncode}\n"); break
    log.write(f"\n===== {time.strftime('%H:%M:%S')} ALL DONE =====\n")
