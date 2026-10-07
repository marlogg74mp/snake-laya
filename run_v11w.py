"""V11w pipeline: dataset (window state + memory) -> fine-tune from v10 -> board-size test. Log: run_v11w.log"""
import subprocess, sys, time
py = sys.executable
steps = [
    ("dataset", [py, "-u", "build_dataset_v11w.py", "70000"]),
    ("train", [py, "-u", "train_v7_full_context.py", "--dataset", "dataset_v11w.jsonl", "--init", "laya_snake_weights_v10",
               "--out", "laya_snake_weights_v11w", "--max-length", "448", "--batch-size", "16", "--grad-accum", "4", "--epochs", "2"]),
    ("eval", [py, "-u", "eval_window.py", "40", "15", "20", "25", "50"]),
]
with open("run_v11w.log", "w", encoding="utf-8") as log:
    for name, cmd in steps:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {name} =====\n"); log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            log.write(f"FAILED with code {r.returncode}\n"); break
    log.write(f"\n===== {time.strftime('%H:%M:%S')} ALL DONE =====\n")
