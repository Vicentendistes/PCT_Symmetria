import argparse
import hashlib
import re
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit overlap between HDF5 training/validation and test splits."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Decimals used by the permutation-invariant exact-cloud hash.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Maximum overlapping values printed for each check.",
    )
    return parser.parse_args()


def base_id(shape_id):
    match = re.match(r"^(\d+)", shape_id)
    return match.group(1) if match else shape_id


def cloud_hash(points, decimals):
    points = np.asarray(points, dtype=np.float64)
    points = np.round(points, decimals=decimals)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    canonical = np.ascontiguousarray(points[order])

    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def load_index(path, decimals):
    import h5py

    names = set()
    base_ids = set()
    hashes = {}

    with h5py.File(path, "r") as h5_file:
        for shape_id in h5_file.keys():
            names.add(shape_id)
            base_ids.add(base_id(shape_id))
            fingerprint = cloud_hash(h5_file[shape_id]["points"][:], decimals)
            hashes.setdefault(fingerprint, []).append(shape_id)

    return {
        "path": path,
        "names": names,
        "base_ids": base_ids,
        "hashes": hashes,
    }


def print_overlap(label, left_values, right_values, show):
    overlap = sorted(left_values & right_values)
    print(f"{label}: {len(overlap)} overlap(s)")
    for value in overlap[:show]:
        print(f"  {value}")
    if len(overlap) > show:
        print(f"  ... and {len(overlap) - show} more")
    return len(overlap)


def compare(reference, test, show):
    print(f"\nReference: {reference['path']}")
    print(f"Test:      {test['path']}")
    print(f"Shapes:    {len(reference['names'])} vs {len(test['names'])}")

    total = 0
    total += print_overlap(
        "Full HDF5 group IDs", reference["names"], test["names"], show
    )
    total += print_overlap(
        "Numeric base IDs", reference["base_ids"], test["base_ids"], show
    )
    total += print_overlap(
        "Exact canonical point hashes",
        set(reference["hashes"]),
        set(test["hashes"]),
        show,
    )
    return total


def main():
    args = parse_args()
    paths = [args.train, args.test]
    if args.valid is not None:
        paths.append(args.valid)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Indexing test split...")
    test = load_index(args.test, args.round_decimals)

    print("Indexing train split...")
    train = load_index(args.train, args.round_decimals)
    overlap_count = compare(train, test, args.show)

    if args.valid is not None:
        print("\nIndexing validation split...")
        valid = load_index(args.valid, args.round_decimals)
        overlap_count += compare(valid, test, args.show)

    print("\nAudit result:")
    if overlap_count == 0:
        print("No overlap detected by the three implemented checks.")
        print("This does not exclude transformed or independently resampled duplicates.")
    else:
        print("Potential overlap detected. Inspect the listed IDs before reporting a holdout test.")


if __name__ == "__main__":
    main()
