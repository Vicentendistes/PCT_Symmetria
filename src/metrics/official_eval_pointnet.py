import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from src.metrics.eval_script import (
    calculate_metrics_from_predictions,
    get_match_sequence_plane_symmetry,
)
from src.model.LightningSantelicesPointNet import (
    LightningSantelicesPointNet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the global Santelices PointNet without dense DBSCAN "
            "post-processing."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--angle-threshold", type=float, default=1.0)
    parser.add_argument("--epsilon-rate", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        default="resultados_evaluacion/pointnet-global",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    model = LightningSantelicesPointNet.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device).eval()

    metric_params = {
        "eps": args.epsilon_rate,
        "theta": 1.0 - math.cos(math.radians(args.angle_threshold)),
        "confidence_threshold": args.confidence_threshold,
        "rot_angle_threshold": math.radians(1.0),
    }

    predictions = []
    predictions_by_category = {}

    with h5py.File(args.test_h5, "r") as h5_file, torch.no_grad():
        for shape_id in tqdm(h5_file.keys(), desc="PointNet global"):
            group = h5_file[shape_id]
            points = torch.as_tensor(
                group["points"][:],
                dtype=torch.float32,
                device=device,
            )
            plane_predictions = model(
                points.unsqueeze(0).transpose(1, 2)
            ).cpu()

            if "planar_symmetries" in group:
                ground_truth = torch.as_tensor(
                    group["planar_symmetries"][:],
                    dtype=torch.float32,
                ).reshape(-1, 6)
            else:
                ground_truth = torch.empty((0, 6), dtype=torch.float32)

            item = (
                points.unsqueeze(0).cpu(),
                plane_predictions,
                [ground_truth],
            )
            predictions.append(item)

            parts = shape_id.split("-")
            category = parts[1] if len(parts) > 1 else "unknown"
            predictions_by_category.setdefault(category, []).append(item)

    mean_ap, phc, _ = calculate_metrics_from_predictions(
        predictions,
        get_match_sequence_plane_symmetry,
        metric_params,
    )

    category_metrics = {}
    for category, category_predictions in sorted(
        predictions_by_category.items()
    ):
        category_map, category_phc, _ = calculate_metrics_from_predictions(
            category_predictions,
            get_match_sequence_plane_symmetry,
            metric_params,
        )
        category_metrics[category] = {
            "mAP": round(float(category_map), 4),
            "PHC": round(float(category_phc), 4),
            "count": len(category_predictions),
        }

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_test": str(Path(args.test_h5).resolve()),
        "parameters": {
            "CONFIDENCE_THRESHOLD": args.confidence_threshold,
            "ANGLE_THRESHOLD": args.angle_threshold,
            "EPSILON_RATE": args.epsilon_rate,
            "POST_PROCESSING": "none (global candidates)",
        },
        "official_metrics": {
            "mAP": round(float(mean_ap), 4),
            "PHC": round(float(phc), 4),
        },
        "metrics_by_category": category_metrics,
    }


def main() -> None:
    args = parse_args()
    results = evaluate(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_path = output_dir / (
        f"eval_mAP{results['official_metrics']['mAP']:.2f}_"
        f"PHC{results['official_metrics']['PHC']:.2f}_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(results["official_metrics"], indent=2))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
