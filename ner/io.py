from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "NER-reproducibility"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def ensure_artifacts(
    root: Path,
    url: str,
    expected_sha256: str,
    archive_name: str = "ner_v35_moe_reproduction_artifacts.zip",
    artifact_version: str = "v35",
) -> Path:
    artifact_root = root / "artifacts" / artifact_version
    marker = artifact_root / "manifest.csv"
    if marker.exists():
        return artifact_root
    archive = root / "artifacts" / archive_name
    if not archive.exists():
        if "RELEASE_URL_PLACEHOLDER" in url:
            raise RuntimeError(
                "The release asset URL has not been configured. "
                "Download the reproduction archive into artifacts/ manually."
            )
        print(f"Downloading frozen reproduction artifacts from {url}")
        download(url, archive)
    actual = sha256(archive)
    if expected_sha256 and "SHA256_PLACEHOLDER" not in expected_sha256:
        if actual != expected_sha256.lower():
            raise RuntimeError(
                f"Artifact SHA256 mismatch: expected {expected_sha256}, got {actual}"
            )
    with zipfile.ZipFile(archive) as compressed:
        compressed.extractall(root / "artifacts")
    if not marker.exists():
        raise RuntimeError(f"Invalid artifact archive: missing {marker}")
    return artifact_root
