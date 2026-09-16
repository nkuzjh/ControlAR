from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MAP_ORDER = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)
MAP_TO_ID = {name: index for index, name in enumerate(MAP_ORDER)}


def read_benchmark_rows(
    data_root: str | Path,
    split: str,
    *,
    max_samples: int | None = None,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    """Read rows through the shared benchmark contract, never by directory scan."""
    try:
        from csgo_benchmark_v2_eval.protocol import BenchmarkData
    except ImportError as exc:
        raise RuntimeError(
            "The shared evaluator protocol is required. Build or sync "
            "csgo_benchmark_v2_eval/protocol.py before running Seen-10."
        ) from exc

    rows = BenchmarkData(str(data_root)).rows(split, maps=list(MAP_ORDER), max_samples=max_samples)
    if not rows:
        raise RuntimeError(f"Benchmark split {split!r} returned no rows from {data_root}")
    for index, row in enumerate(rows):
        missing = {"sample_id", "map_name", "file_frame", "image_path", "radar_path", "pose"} - row.keys()
        if missing:
            raise ValueError(f"Benchmark row {index} is missing fields: {sorted(missing)}")
        if row["map_name"] not in MAP_TO_ID:
            raise ValueError(f"Unexpected map in Seen-10 split: {row['map_name']!r}")
        if len(row["pose"]) != 5 or not np.isfinite(np.asarray(row["pose"], dtype=np.float32)).all():
            raise ValueError(f"Invalid normalized pose for sample {row['sample_id']}: {row['pose']!r}")
        if not Path(row["radar_path"]).is_file():
            raise FileNotFoundError(f"Missing radar for sample {row['sample_id']}: {row['radar_path']}")
        if require_images and not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Missing benchmark frame for sample {row['sample_id']}: {row['image_path']}")
    return rows


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor.div_(127.5).sub_(1.0)


class Seen10GenerationDataset(Dataset):
    """Manifest-driven paired data. In inference mode target images are never opened."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        image_size: int = 448,
        include_target: bool = True,
    ) -> None:
        if image_size % 16:
            raise ValueError(f"image_size must be divisible by the VQ stride 16, got {image_size}")
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.include_target = bool(include_target)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["radar_path"]) as opened:
            radar = opened.convert("RGB").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
            )

        item: dict[str, Any] = {
            "radar": _to_normalized_tensor(radar),
            "pose": torch.as_tensor(row["pose"], dtype=torch.float32),
            "map_id": torch.tensor(MAP_TO_ID[row["map_name"]], dtype=torch.long),
            "sample_id": row["sample_id"],
            "map_name": row["map_name"],
            "file_frame": row["file_frame"],
        }
        if row.get("clip_id") is not None:
            item["clip_id"] = row["clip_id"]
            item["frame_index"] = row["frame_index"]
        if self.include_target:
            with Image.open(row["image_path"]) as opened:
                target = opened.convert("RGB").resize(
                    (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
                )
            item["target"] = _to_normalized_tensor(target)
        return item
