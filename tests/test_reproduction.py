from __future__ import annotations

from pathlib import Path

from reproduce import reproduce, verify


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_main_result(tmp_path: Path) -> None:
    artifact_root = ROOT / "artifacts" / "v35"
    if not (artifact_root / "manifest.csv").exists():
        return
    overall = reproduce(artifact_root, tmp_path)
    verify(overall, tolerance=1e-8)
