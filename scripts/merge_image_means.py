"""Merge equally sized image-mean shards into one average.

By default this reads:
    image_means/ViT-L-14_count_10_shard-{0..5}-of-6.pt

The shard files may contain either a raw tensor or the partition payload saved
by libs/SITH/scripts/compute_image_mean.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("image_means"),
        help="Directory containing the shard .pt files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ViT-L-14_count_10",
        help="Filename prefix before _shard-X-of-N.pt.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=6,
        help="Number of equally sized shards to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_means/ViT-L-14_count_10.pt"),
        help="Path where the merged average tensor will be saved.",
    )
    return parser


def load_mean(path: Path) -> torch.Tensor:
    value: Any = torch.load(path, map_location="cpu")

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, dict) and isinstance(value.get("mean"), torch.Tensor):
        return value["mean"]

    msg = f"{path} must contain a tensor or a dict with a tensor under 'mean'."
    raise TypeError(msg)


def main() -> None:
    args = get_parser().parse_args()

    if args.num_shards <= 0:
        msg = "--num-shards must be positive."
        raise ValueError(msg)

    shard_paths = [
        args.input_dir / f"{args.prefix}_shard-{idx}-of-{args.num_shards}.pt"
        for idx in range(args.num_shards)
    ]

    means = []
    output_dtype: torch.dtype | None = None
    expected_shape: torch.Size | None = None
    for path in shard_paths:
        if not path.is_file():
            msg = f"Missing shard: {path}"
            raise FileNotFoundError(msg)
        mean = load_mean(path)
        if expected_shape is None:
            expected_shape = mean.shape
        elif mean.shape != expected_shape:
            msg = f"{path} has shape {tuple(mean.shape)}, expected {tuple(expected_shape)}."
            raise ValueError(msg)
        output_dtype = output_dtype or mean.dtype
        means.append(mean.to(dtype=torch.float64))

    if output_dtype is None:
        msg = "No shard means were loaded."
        raise RuntimeError(msg)

    merged = torch.stack(means, dim=0).mean(dim=0).to(dtype=output_dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)

    print(f"Saved merged mean to {args.output}")


if __name__ == "__main__":
    main()
