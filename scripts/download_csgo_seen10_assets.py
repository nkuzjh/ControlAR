#!/usr/bin/env python3
"""Fetch pinned official ControlAR/DINOv2 weights with resumable HTTP ranges."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 32 * 1024 * 1024
WORKERS = max(1, int(os.environ.get("CSGO_ASSET_DOWNLOAD_WORKERS", "4")))
READ_BYTES = 1024 * 1024

ASSETS = [
    {
        "repo": "wondervictor/ControlAR",
        "revision": "22cecd7a873db8df97ae2b2dc88befee72e97a3a",
        "filename": "canny_MR.safetensors",
        "size": 3_356_608_032,
        "sha256": "ef59b3c51e582e4742406480fb81160044b902bd46b2f00d923734800258545e",
        "target": ROOT / "checkpoints/t2i/canny_MR.safetensors",
        "part_dir": ROOT / "checkpoints/t2i/.canny_MR_parts",
    },
    {
        "repo": "peizesun/llamagen_t2i",
        "revision": "276f5c5a3d915b922899a03f1912605531574747",
        "filename": "vq_ds16_t2i.pt",
        "size": 287_920_306,
        "sha256": "0e21fc1318e2e9ee641a07bdad0e20675e9ec35e6e3eb911d58b5d7a2cd8d4cb",
        "target": ROOT / "checkpoints/vq/vq_ds16_t2i.pt",
        "part_dir": ROOT / "checkpoints/vq/.vq_ds16_t2i_parts",
    },
    {
        "repo": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "filename": "model.safetensors",
        "size": 88_249_960,
        "sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "target": ROOT / "autoregressive/models/dinov2-small/model.safetensors",
        "part_dir": ROOT / "autoregressive/models/dinov2-small/.model_safetensors_parts",
    },
]

# Official Git blob IDs from the pinned facebook/dinov2-small revision.
DINO_SMALL_FILES = {
    "config.json": "5664b325e6258d3960fad8c4c1cff958f3cc2272",
    "preprocessor_config.json": "ff5b47c2edcd1d3556d63c01a65d93b58b9efce1",
}

URLS = {
    (asset["repo"], asset["filename"]): (
        f"https://huggingface.co/{asset['repo']}/resolve/"
        f"{asset['revision']}/{asset['filename']}?download=true"
    )
    for asset in ASSETS
}
LOCAL = threading.local()
PROGRESS_LOCK = threading.Lock()
PROGRESS = {"done": 0, "downloaded": 0, "next_percent": 20}


def get_session() -> requests.Session:
    session = getattr(LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        LOCAL.session = session
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def asset_url(asset: dict) -> str:
    return URLS[(asset["repo"], asset["filename"])]


def expected_part_size(asset: dict, index: int) -> int:
    start = index * CHUNK_BYTES
    return min(CHUNK_BYTES, asset["size"] - start)


def part_path(asset: dict, index: int) -> Path:
    return asset["part_dir"] / f"{index:05d}.part"


def import_manual_canny_ranges() -> None:
    """Keep the already measured official ranges and resume them in place."""
    asset = ASSETS[0]
    part_dir = asset["part_dir"]
    part_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        canonical = part_path(asset, index)
        if canonical.exists():
            continue
        if index == 0:
            source_parts = [
                part_dir / "part-0000",
                part_dir / "part-0000-rest",
            ]
        else:
            source_parts = [part_dir / f"part-{index:04d}"]
        if not all(path.is_file() for path in source_parts):
            continue
        if sum(path.stat().st_size for path in source_parts) != expected_part_size(
            asset, index
        ):
            continue
        with canonical.open("wb") as target:
            for source in source_parts:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
        print(
            f"Reused tested official Canny range {index + 1}/5 "
            f"({canonical.stat().st_size:,} bytes)",
            flush=True,
        )


def migrate_hf_incomplete_dino() -> None:
    """Reuse a partial regular-HTTP DINO transfer if setup was interrupted."""
    asset = ASSETS[2]
    target = asset["target"]
    if target.exists():
        return
    incomplete_dir = target.parent / ".cache/huggingface/download"
    if not incomplete_dir.is_dir():
        return
    candidates = list(incomplete_dir.glob("*.ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1.incomplete"))
    if not candidates:
        return
    source = candidates[0]
    part_dir = asset["part_dir"]
    part_dir.mkdir(parents=True, exist_ok=True)
    total = source.stat().st_size
    offset = 0
    index = 0
    with source.open("rb") as stream:
        while offset < total:
            part = part_path(asset, index)
            capacity = expected_part_size(asset, index)
            if not part.exists():
                amount = min(capacity, total - offset)
                with part.open("wb") as output:
                    shutil.copyfileobj(_LimitedReader(stream, amount), output)
            else:
                stream.seek(capacity, os.SEEK_CUR)
            offset += capacity
            index += 1
    print(f"Reused {total:,} bytes from an interrupted regular-HTTP DINO transfer", flush=True)


class _LimitedReader:
    def __init__(self, source, remaining: int) -> None:
        self.source = source
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunk = self.source.read(size)
        self.remaining -= len(chunk)
        return chunk


def existing_bytes(asset: dict) -> int:
    target = asset["target"]
    if target.exists():
        actual = sha256_file(target)
        if actual != asset["sha256"]:
            raise RuntimeError(
                f"Refusing to overwrite existing {target}: expected SHA256 "
                f"{asset['sha256']}, got {actual}"
            )
        return asset["size"]

    total = 0
    count = (asset["size"] + CHUNK_BYTES - 1) // CHUNK_BYTES
    for index in range(count):
        part = part_path(asset, index)
        if not part.exists():
            continue
        size = part.stat().st_size
        if size > expected_part_size(asset, index):
            raise RuntimeError(f"Unexpected oversized download part: {part} ({size})")
        total += size
    return total


def update_progress(amount: int, overall_size: int, started: float) -> None:
    with PROGRESS_LOCK:
        PROGRESS["done"] += amount
        PROGRESS["downloaded"] += amount
        percent = 100 * PROGRESS["done"] / overall_size
        while percent >= PROGRESS["next_percent"]:
            elapsed = max(time.monotonic() - started, 0.001)
            speed = PROGRESS["downloaded"] / elapsed
            remaining = max(overall_size - PROGRESS["done"], 0)
            eta_seconds = remaining / speed if speed else 0
            eta_minutes = eta_seconds / 60
            print(
                f"Asset transfer {PROGRESS['next_percent']}% "
                f"({PROGRESS['done']:,}/{overall_size:,} bytes), "
                f"{speed / 1_000_000:.2f} MB/s, ETA {eta_minutes:.1f} min",
                flush=True,
            )
            PROGRESS["next_percent"] += 20


def download_part(asset: dict, index: int, overall_size: int, started: float) -> None:
    # Another invocation may have completed and verified this asset while a
    # queued range was waiting. Do not continue transferring its leftover parts.
    if asset["target"].exists():
        return
    part = part_path(asset, index)
    part.parent.mkdir(parents=True, exist_ok=True)
    chunk_start = index * CHUNK_BYTES
    chunk_end = chunk_start + expected_part_size(asset, index) - 1
    retries = 0

    while True:
        current_size = part.stat().st_size if part.exists() else 0
        expected_size = expected_part_size(asset, index)
        if current_size > expected_size:
            raise RuntimeError(f"Oversized partial range: {part} ({current_size})")
        if current_size == expected_size:
            return

        request_start = chunk_start + current_size
        headers = {
            "Range": f"bytes={request_start}-{chunk_end}",
            "Accept-Encoding": "identity",
        }
        try:
            with get_session().get(
                asset_url(asset),
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range request returned HTTP {response.status_code} for "
                        f"{asset['repo']}/{asset['filename']} bytes "
                        f"{request_start}-{chunk_end}"
                    )
                expected_header = f"bytes {request_start}-{chunk_end}/{asset['size']}"
                if response.headers.get("Content-Range") != expected_header:
                    raise RuntimeError(
                        f"Unexpected Content-Range for {asset['filename']}: "
                        f"{response.headers.get('Content-Range')!r}, expected {expected_header!r}"
                    )
                with part.open("ab") as output:
                    for payload in response.iter_content(READ_BYTES):
                        if not payload:
                            continue
                        output.write(payload)
                        update_progress(len(payload), overall_size, started)
                    output.flush()
                    os.fsync(output.fileno())
            if part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Short range for {part}: {part.stat().st_size}/{expected_size} bytes"
                )
            return
        except Exception as exc:
            retries += 1
            if retries > 6:
                raise
            delay = min(2**retries, 30)
            print(
                f"Retry {retries}/6 {asset['filename']} range {index + 1}: "
                f"{type(exc).__name__}: {exc}; keeping partial bytes, wait {delay}s",
                flush=True,
            )
            time.sleep(delay)


def assemble_asset(asset: dict) -> None:
    target = asset["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".assembling")
    count = (asset["size"] + CHUNK_BYTES - 1) // CHUNK_BYTES
    with temporary.open("wb") as output:
        for index in range(count):
            part = part_path(asset, index)
            expected = expected_part_size(asset, index)
            actual = part.stat().st_size if part.exists() else 0
            if actual != expected:
                raise RuntimeError(f"Incomplete range {part}: {actual}/{expected} bytes")
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())

    if temporary.stat().st_size != asset["size"]:
        raise RuntimeError(f"Assembled size mismatch for {temporary}")
    actual = sha256_file(temporary)
    if actual != asset["sha256"]:
        raise RuntimeError(
            f"Official SHA256 mismatch for {asset['filename']}: "
            f"expected {asset['sha256']}, got {actual}; kept ranges in {asset['part_dir']}"
        )
    os.replace(temporary, target)
    print(
        f"Verified {asset['repo']}@{asset['revision']}/{asset['filename']} "
        f"-> {target} ({asset['size']:,} bytes, sha256={actual})",
        flush=True,
    )
    shutil.rmtree(asset["part_dir"], ignore_errors=True)


def fetch_dino_metadata() -> None:
    repo = "facebook/dinov2-small"
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    destination = ROOT / "autoregressive/models/dinov2-small"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected_blob in DINO_SMALL_FILES.items():
        target = destination / filename
        if target.exists():
            data = target.read_bytes()
        else:
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"
            with get_session().get(url, timeout=(30, 120)) as response:
                response.raise_for_status()
                data = response.content
            if git_blob_id(data) != expected_blob:
                raise RuntimeError(
                    f"Pinned HF Git blob mismatch for {filename}: "
                    f"expected {expected_blob}, got {git_blob_id(data)}"
                )
            temporary = target.with_name(target.name + ".partial")
            temporary.write_bytes(data)
            os.replace(temporary, target)
        actual_blob = git_blob_id(data)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"Existing pinned DINO file {target} has Git blob {actual_blob}, "
                f"expected {expected_blob}"
            )
        print(f"Verified {repo}@{revision}/{filename} -> {target}", flush=True)


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import_manual_canny_ranges()
    migrate_hf_incomplete_dino()

    overall_size = sum(asset["size"] for asset in ASSETS)
    initial = sum(existing_bytes(asset) for asset in ASSETS)
    PROGRESS["done"] = initial
    PROGRESS["next_percent"] = 20 * (initial // (overall_size // 5) + 1)
    started = time.monotonic()
    print(
        f"Starting pinned model downloads: {initial:,}/{overall_size:,} bytes "
        f"already present; {WORKERS} parallel HTTP Range workers",
        flush=True,
    )

    jobs_by_asset = []
    for asset in ASSETS:
        if asset["target"].exists():
            continue
        count = (asset["size"] + CHUNK_BYTES - 1) // CHUNK_BYTES
        asset_jobs = []
        for index in range(count):
            part = part_path(asset, index)
            if not part.exists() or part.stat().st_size < expected_part_size(asset, index):
                asset_jobs.append((asset, index))
        jobs_by_asset.append(asset_jobs)

    # Interleave model ranges by index. This starts the much smaller VQ and
    # DINO transfers alongside Canny instead of waiting behind all Canny ranges.
    jobs = []
    for index in range(max((len(group) for group in jobs_by_asset), default=0)):
        for group in jobs_by_asset:
            if index < len(group):
                jobs.append(group[index])

    remaining = {asset["filename"]: 0 for asset in ASSETS}
    for asset, _ in jobs:
        remaining[asset["filename"]] += 1

    # If an earlier run completed every range but stopped before assembly,
    # verify that asset now instead of waiting for unrelated downloads.
    for asset in ASSETS:
        if not asset["target"].exists() and remaining[asset["filename"]] == 0:
            part_count = (asset["size"] + CHUNK_BYTES - 1) // CHUNK_BYTES
            if any(part_path(asset, i).exists() for i in range(part_count)):
                assemble_asset(asset)

    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {}
            for asset, index in jobs:
                future = executor.submit(
                    download_part, asset, index, overall_size, started
                )
                futures[future] = asset
            for future in as_completed(futures):
                asset = futures[future]
                future.result()
                filename = asset["filename"]
                remaining[filename] -= 1
                if remaining[filename] == 0 and not asset["target"].exists():
                    assemble_asset(asset)

    for asset in ASSETS:
        if not asset["target"].exists():
            assemble_asset(asset)
        else:
            actual = sha256_file(asset["target"])
            if actual != asset["sha256"]:
                raise RuntimeError(f"Official SHA256 mismatch for {asset['target']}: {actual}")
            print(f"Verified existing {asset['target']} sha256={actual}", flush=True)
            shutil.rmtree(asset["part_dir"], ignore_errors=True)

    fetch_dino_metadata()


if __name__ == "__main__":
    main()
