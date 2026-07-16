from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        [sys.executable, str(ROOT / "reproduce.py")],
        [sys.executable, str(ROOT / "exp/component_ablation.py")],
        [sys.executable, str(ROOT / "exp/operating_sensitivity.py")],
        [sys.executable, str(ROOT / "exp/center_to_interval.py")],
        [sys.executable, str(ROOT / "exp/random_insertion_guardrail.py")],
        [sys.executable, str(ROOT / "exp/evidence_stress.py")],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
