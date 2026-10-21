"""Smoke test: generate toy data, run every stage quickly. Usage: python scripts/00_setup.py"""
import subprocess, sys
stages = [
    [sys.executable, "scripts/02_prepare_lstm_dataset.py", "--toy"],
    [sys.executable, "scripts/03_train_lstm.py", "--toy", "--epochs", "2"],
    [sys.executable, "scripts/04_prepare_extent_dataset.py", "--toy"],
    [sys.executable, "scripts/05_train_extent_cnn.py", "--epochs", "2"],
    [sys.executable, "scripts/06_evaluate.py"],
]
for s in stages:
    print(">>>", " ".join(s)); subprocess.run(s, check=True)
print("Pipeline smoke test PASSED")
