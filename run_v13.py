"""V13 pipeline: dataset with exact apple coords -> fine-tune from v12 -> DAgger round -> fine-tune -> eval.
Log: run_v13.log"""
import subprocess, sys, time
py = sys.executable
train = lambda data, init, out, epochs: [py, "-u", "train_v7_full_context.py", "--dataset", data, "--init", init, "--out", out,
                                         "--max-length", "512", "--batch-size", "16", "--grad-accum", "4", "--epochs", epochs]
steps = [
    ("dataset", [py, "-u", "build_dataset_v13.py", "data", "80000"]),
    ("train v13", train("dataset_v13.jsonl", "laya_snake_weights_v12", "laya_snake_weights_v13", "2")),
    ("dagger", [py, "-u", "build_dataset_v13.py", "dagger", "40000", "laya_snake_weights_v13"]),
    ("train v13d", train("dataset_v13d.jsonl", "laya_snake_weights_v13", "laya_snake_weights_v13d", "1")),
    ("eval", [py, "-u", "eval_v13.py", "100", "20", "30"]),
]
with open("run_v13.log", "w", encoding="utf-8") as log:
    for name, cmd in steps:
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {name} =====\n"); log.flush()
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            log.write(f"FAILED with code {r.returncode}\n"); break
    log.write(f"\n===== {time.strftime('%H:%M:%S')} ALL DONE =====\n")
