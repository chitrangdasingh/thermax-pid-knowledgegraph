"""Clone the two upstream Microsoft reference repositories without modifying them."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPOSITORIES = {
    "MLOpsManufacturing": "https://github.com/Azure-Samples/MLOpsManufacturing.git",
    "digitization-of-piping-and-instrument-diagrams": "https://github.com/Azure-Samples/digitization-of-piping-and-instrument-diagrams.git",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="external")
    parser.add_argument("--ref", default="main")
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in REPOSITORIES.items():
        target = destination / name
        if target.exists():
            print(f"SKIP {target}: already exists")
            continue
        subprocess.run(["git", "clone", "--depth", "1", "--branch", args.ref, url, str(target)], check=True)
    training = destination / "MLOpsManufacturing" / "samples" / "amlv2_pid_symbol_detection_train"
    inference = destination / "digitization-of-piping-and-instrument-diagrams"
    if not training.is_dir() or not (inference / "docs" / "user-guide.md").is_file():
        raise RuntimeError("Expected Microsoft sample paths were not found; inspect the selected upstream revision")
    print("Training sample:", training)
    print("Inference sample:", inference)


if __name__ == "__main__":
    main()
