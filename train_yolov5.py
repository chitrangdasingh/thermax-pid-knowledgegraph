"""Launch official YOLOv5 training only after the dataset quality gate passes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from validate_yolov5_dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml")
    parser.add_argument("--yolov5-repo", required=True, help="Local clone of https://github.com/ultralytics/yolov5")
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runtime/yolov5_runs")
    parser.add_argument("--name", default="thermax_pid_symbols")
    args = parser.parse_args()

    report = validate_dataset(args.data_yaml)
    repo = Path(args.yolov5_repo).resolve()
    train_script = repo / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Official YOLOv5 train.py not found in {repo}")
    print(json.dumps(report, indent=2))
    command = [
        sys.executable, str(train_script), "--data", str(Path(args.data_yaml).resolve()),
        "--weights", args.weights, "--epochs", str(args.epochs), "--img", str(args.img),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--project", str(Path(args.project).resolve()), "--name", args.name,
        "--exist-ok", "--cache", "ram",
    ]
    print("Launching:", " ".join(command))
    subprocess.run(command, cwd=repo, check=True)
    print("Training finished. Run val.py on the untouched test split and obtain engineering sign-off before deployment.")


if __name__ == "__main__":
    main()
