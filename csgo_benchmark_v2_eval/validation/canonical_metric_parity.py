#!/usr/bin/env python3
"""Compare this evaluator's metrics with AST-extracted UniLIP functions.

All image fixtures are generated under a temporary directory and removed at
exit. No benchmark data or prediction outputs are read or written.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import sys
import tempfile
import typing
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

import metrics_continuous as evaluator_continuous  # noqa: E402
import metrics_images as evaluator_images  # noqa: E402


FVD_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"


def _extract_definitions(path: Path, names: set[str], scope: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in nodes}
    if found != names:
        raise RuntimeError(f"Missing canonical definitions in {path}: {sorted(names - found)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)


def _cpu_loader(dataset: Dataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    del num_workers
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def _metric_ast_equal(left: Path, right: Path, names: tuple[str, ...]) -> dict[str, bool]:
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (left, right)]
    defs = []
    for tree in trees:
        defs.append({
            node.name: ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        })
    return {name: defs[0].get(name) == defs[1].get(name) for name in names}


def _make_fixture(
    root: Path,
    frame_record_cls: type,
) -> tuple[Path, Path, list[str], list[dict[str, Any]], list[list[Any]]]:
    gt_dir = root / "gt"
    pred_root = root / "pred"
    pred_dir = pred_root / "cs_agency"
    gt_dir.mkdir()
    pred_dir.mkdir(parents=True)
    random = np.random.default_rng(1907)
    names: list[str] = []
    evaluator_tracks: list[dict[str, Any]] = []
    canonical_tracks: list[list[Any]] = []
    yy, xx = np.mgrid[0:40, 0:40]

    for clip_index in range(2):
        evaluator_rows = []
        canonical_records = []
        for frame_index in range(64):
            filename = f"clip{clip_index}_frame_{frame_index:03d}.jpg"
            sample = np.stack(
                [
                    (xx * 4 + frame_index * (clip_index + 1)) % 256,
                    (yy * 5 + frame_index * 2 + clip_index * 13) % 256,
                    (xx * 2 + yy * 3 + frame_index * 3) % 256,
                ],
                axis=-1,
            ).astype(np.uint8)
            noise = random.integers(-8, 9, size=sample.shape, dtype=np.int16)
            predicted = np.clip(
                np.roll(sample, shift=(frame_index % 3) - 1, axis=1).astype(np.int16) + noise,
                0,
                255,
            ).astype(np.uint8)
            Image.fromarray(sample).save(gt_dir / filename, quality=92)
            Image.fromarray(predicted).save(pred_dir / filename, quality=92)
            names.append(filename)
            evaluator_rows.append(
                {
                    "map_name": "cs_agency",
                    "file_frame": filename[:-4],
                    "image_path": str(gt_dir / filename),
                    "file_num": clip_index + 1,
                    "frame_id": 1000 + frame_index,
                }
            )
            canonical_records.append(
                frame_record_cls(
                    filename=filename,
                    file_num=clip_index + 1,
                    frame_id=1000 + frame_index,
                )
            )
        evaluator_tracks.append(
            {
                "clip_id": f"clip{clip_index}",
                "map_name": "cs_agency",
                "rows": evaluator_rows,
            }
        )
        canonical_tracks.append(canonical_records)
    return gt_dir, pred_root, names, evaluator_tracks, canonical_tracks


def _run(
    unilip_root: Path,
    report_path: Path,
    *,
    run_fvd_smoke: bool,
    fvd_device: str | None,
) -> dict[str, Any]:
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    image_source = unilip_root / "benchmark_csgo_v1.py"
    continuous_source = unilip_root / "benchmark_csgo_v1_conti.py"
    upstream_fvd = unilip_root / "third_party/PyTorch-Frechet-Video-Distance/fvd_metric/fvd.py"
    local_fvd = EVAL_DIR / "fvd_metric/fvd.py"

    canonical_image: dict[str, Any] = {
        "os": os,
        "Path": Path,
        "Sequence": typing.Sequence,
        "Dataset": Dataset,
        "DataLoader": _cpu_loader,
        "Image": Image,
        "transforms": transforms,
        "InterpolationMode": InterpolationMode,
        "torch": torch,
        "F": F,
        "PeakSignalNoiseRatio": PeakSignalNoiseRatio,
        "StructuralSimilarityIndexMeasure": StructuralSimilarityIndexMeasure,
        "FrechetInceptionDistance": FrechetInceptionDistance,
        "tqdm": lambda items, **kwargs: items,
    }
    _extract_definitions(
        image_source,
        {
            "PairedImageDataset",
            "compute_psnr_ssim",
            "_sobel_edge_map",
            "compute_boundary_metrics",
            "SingleImageDataset",
            "compute_fid",
        },
        canonical_image,
    )
    canonical_continuous: dict[str, Any] = {
        "os": os,
        "Sequence": typing.Sequence,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Dict": typing.Dict,
        "np": np,
        "Image": Image,
        "dataclass": dataclasses.dataclass,
        "tqdm": lambda items, **kwargs: items,
    }
    _extract_definitions(
        continuous_source,
        {
            "FrameRecord",
            "load_rgb_uint8",
            "load_track_pair",
            "mean_or_none",
            "require_cv2",
            "estimate_farneback_flow",
            "warp_with_flow",
            "compute_temporal_metrics",
        },
        canonical_continuous,
    )

    source_equal = _metric_ast_equal(
        upstream_fvd,
        local_fvd,
        ("get_feature_detector", "compute_feature_stats", "compute_fvd"),
    )
    if not all(source_equal.values()):
        raise AssertionError(f"Copied FVD algorithm differs from upstream: {source_equal}")

    with tempfile.TemporaryDirectory(prefix="canonical_metric_parity_", dir=EVAL_DIR) as tmp:
        gt_dir, pred_root, names, evaluator_tracks, canonical_tracks = _make_fixture(
            Path(tmp), canonical_continuous["FrameRecord"]
        )
        pred_dir = pred_root / "cs_agency"

        canonical_psnr, canonical_ssim = canonical_image["compute_psnr_ssim"](
            str(gt_dir), str(pred_dir), names[:5], 32, 2, "cpu"
        )
        evaluator_psnr, evaluator_ssim = evaluator_images.compute_psnr_ssim(
            [gt_dir / name for name in names[:5]],
            [pred_dir / name for name in names[:5]],
            size=32,
            batch_size=2,
            device="cpu",
            num_workers=0,
        )
        canonical_boundary = canonical_image["compute_boundary_metrics"](
            str(gt_dir), str(pred_dir), names[:5], 32, 2, "cpu", 0.85, 2
        )["Boundary_F1"]
        evaluator_boundary = evaluator_images.compute_boundary_metrics(
            [gt_dir / name for name in names[:5]],
            [pred_dir / name for name in names[:5]],
            size=32,
            batch_size=2,
            device="cpu",
            edge_quantile=0.85,
            edge_tolerance=2,
            num_workers=0,
        )["Boundary_F1"]

        canonical_temporal = canonical_continuous["compute_temporal_metrics"](
            str(gt_dir), str(pred_dir), canonical_tracks, 24
        )
        evaluator_temporal = evaluator_continuous.compute_temporal_metrics(
            evaluator_tracks, pred_root, size=24
        )

        canonical_fid = canonical_image["compute_fid"](
            str(gt_dir), str(pred_dir), names[:5], 2, "cpu"
        )
        evaluator_fid = evaluator_images.compute_fid(
            [gt_dir / name for name in names[:5]],
            [pred_dir / name for name in names[:5]],
            batch_size=2,
            device="cpu",
            num_workers=0,
        )

    metrics = {
        "PSNR": {"canonical": float(canonical_psnr), "evaluator": float(evaluator_psnr)},
        "SSIM": {"canonical": float(canonical_ssim), "evaluator": float(evaluator_ssim)},
        "Boundary_F1": {"canonical": float(canonical_boundary), "evaluator": float(evaluator_boundary)},
        "TWE": {
            "canonical": float(canonical_temporal["Temporal_Warping_Error"]),
            "evaluator": float(evaluator_temporal["TWE"]),
        },
        "TDE": {
            "canonical": float(canonical_temporal["Temporal_Difference_Error"]),
            "evaluator": float(evaluator_temporal["TDE"]),
        },
        "FID": {"canonical": float(canonical_fid), "evaluator": float(evaluator_fid)},
    }
    for item in metrics.values():
        item["absolute_delta"] = abs(item["canonical"] - item["evaluator"])
        if not math.isclose(item["canonical"], item["evaluator"], rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(f"Metric parity failed: {item}")

    cache_dir = unilip_root / "loaded_models"
    os.environ["UNILIP_FVD_CACHE_DIR"] = str(cache_dir)
    from fvd_metric.fvd import get_feature_detector

    detector = get_feature_detector(FVD_URL, "cpu")
    report = {
        "check": "synthetic canonical metric parity",
        "status": "pass",
        "fixture": "2 synthetic 64-frame tracks; 40x40 JPEG inputs; all temporary frames removed",
        "canonical_sources": {
            "image_metrics": str(image_source),
            "continuous_metrics": str(continuous_source),
            "fvd_upstream": str(upstream_fvd),
        },
        "metrics": metrics,
        "fvd_algorithm_ast_identical": source_equal,
        "fvd_i3d_weight_cache": str(cache_dir),
        "fvd_i3d_torchscript_loaded": True,
    }
    if run_fvd_smoke:
        from fvd_metric import compute_fvd

        selected_device = fvd_device or ("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device="cpu").manual_seed(2026)
        true_clips = torch.randint(
            0,
            256,
            (2, 3, 16, 224, 224),
            dtype=torch.uint8,
            generator=generator,
        )
        pred_clips = torch.roll(true_clips, shifts=1, dims=-1)
        score = compute_fvd(
            true_clips,
            pred_clips,
            max_items=2,
            device=torch.device(selected_device),
            batch_size=1,
        )
        if not math.isfinite(score):
            raise AssertionError(f"Synthetic two-clip FVD returned a non-finite value: {score}")
        report["fvd_synthetic_smoke"] = {
            "status": "pass",
            "device": selected_device,
            "clips_per_side": 2,
            "frames_per_clip": 16,
            "frame_size": 224,
            "score": float(score),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unilip-root", type=Path, default=Path("/home/jiahao/task/UniLIP"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "canonical_metric_parity_report.json",
    )
    parser.add_argument(
        "--run-fvd-smoke",
        action="store_true",
        help="Run actual FVD on two synthetic 16-frame 224x224 clips per side",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    report = _run(
        args.unilip_root.expanduser().resolve(),
        args.report.expanduser().resolve(),
        run_fvd_smoke=args.run_fvd_smoke,
        fvd_device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
