#!/usr/bin/env python3
"""Compare ordered SITH interpretations across two result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DIR = Path("results/ViT_L-14/base")
DEFAULT_UPDATED_DIR = Path("results/ViT_L-14/base_updated")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help=f"Directory containing reference JSON files. Default: {DEFAULT_REFERENCE_DIR}",
    )
    parser.add_argument(
        "--updated-dir",
        type=Path,
        default=DEFAULT_UPDATED_DIR,
        help=f"Directory containing updated JSON files. Default: {DEFAULT_UPDATED_DIR}",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["concept", "index", "score"],
        default=["index", "concept"],
        help="Entry fields to compare in order. Use 'score' too for exact score matches.",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=20,
        help="Maximum number of mismatched singular vectors to print.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable comparison report.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    return data


def relative_json_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*.json")
        if path.is_file()
    }


def comparable_entries(entries: Any, fields: list[str]) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        msg = f"Expected singular-vector interpretation to be a list, got {type(entries)}."
        raise TypeError(msg)
    return [
        {field: entry.get(field) for field in fields}
        for entry in entries
        if isinstance(entry, dict)
    ]


def iter_singular_vectors(data: dict[str, Any]) -> list[tuple[str, str, Any]]:
    rows: list[tuple[str, str, Any]] = []
    for head in sorted(data):
        svs = data[head]
        if not isinstance(svs, dict):
            msg = f"Expected {head} to contain a singular-vector object."
            raise TypeError(msg)
        for sv in sorted(svs):
            rows.append((head, sv, svs[sv]))
    return rows


def compare_file(
    reference_path: Path,
    updated_path: Path,
    rel_path: Path,
    fields: list[str],
) -> tuple[list[dict[str, Any]], int]:
    reference = load_json(reference_path)
    updated = load_json(updated_path)

    reference_keys = {(head, sv) for head, sv, _ in iter_singular_vectors(reference)}
    updated_keys = {(head, sv) for head, sv, _ in iter_singular_vectors(updated)}

    mismatches: list[dict[str, Any]] = []
    for head, sv in sorted(reference_keys - updated_keys):
        mismatches.append(
            {
                "file": str(rel_path),
                "head": head,
                "singular_vector": sv,
                "reason": "missing_in_updated",
            }
        )
    for head, sv in sorted(updated_keys - reference_keys):
        mismatches.append(
            {
                "file": str(rel_path),
                "head": head,
                "singular_vector": sv,
                "reason": "missing_in_reference",
            }
        )

    compared = 0
    for head, sv in sorted(reference_keys & updated_keys):
        reference_entries = comparable_entries(reference[head][sv], fields)
        updated_entries = comparable_entries(updated[head][sv], fields)
        compared += 1
        if reference_entries == updated_entries:
            continue
        mismatches.append(
            {
                "file": str(rel_path),
                "head": head,
                "singular_vector": sv,
                "reason": "different_interpretation",
                "reference": reference_entries,
                "updated": updated_entries,
            }
        )

    return mismatches, compared


def main() -> None:
    args = get_parser().parse_args()

    reference_files = relative_json_files(args.reference_dir)
    updated_files = relative_json_files(args.updated_dir)

    missing_in_updated = sorted(reference_files - updated_files)
    missing_in_reference = sorted(updated_files - reference_files)
    common_files = sorted(reference_files & updated_files)

    mismatches: list[dict[str, Any]] = []
    for rel_path in missing_in_updated:
        mismatches.append({"file": str(rel_path), "reason": "missing_file_in_updated"})
    for rel_path in missing_in_reference:
        mismatches.append(
            {"file": str(rel_path), "reason": "missing_file_in_reference"}
        )

    compared_vectors = 0
    for rel_path in common_files:
        file_mismatches, file_compared = compare_file(
            args.reference_dir / rel_path,
            args.updated_dir / rel_path,
            rel_path,
            args.fields,
        )
        mismatches.extend(file_mismatches)
        compared_vectors += file_compared

    matched_vectors = compared_vectors - sum(
        1 for item in mismatches if item.get("reason") == "different_interpretation"
    )

    report = {
        "reference_dir": str(args.reference_dir),
        "updated_dir": str(args.updated_dir),
        "fields": args.fields,
        "files_compared": len(common_files),
        "missing_files_in_updated": [str(path) for path in missing_in_updated],
        "missing_files_in_reference": [str(path) for path in missing_in_reference],
        "singular_vectors_compared": compared_vectors,
        "singular_vectors_matched": matched_vectors,
        "singular_vectors_different": compared_vectors - matched_vectors,
        "mismatches": mismatches,
    }

    print("Interpretation comparison:")
    print(f"  reference_dir: {args.reference_dir}")
    print(f"  updated_dir: {args.updated_dir}")
    print(f"  fields: {', '.join(args.fields)}")
    print(f"  files compared: {len(common_files)}")
    print(f"  singular vectors compared: {compared_vectors}")
    print(f"  singular vectors matched: {matched_vectors}")
    print(f"  singular vectors different: {compared_vectors - matched_vectors}")
    print(f"  missing files in updated: {len(missing_in_updated)}")
    print(f"  missing files in reference: {len(missing_in_reference)}")

    if mismatches:
        print("\nMismatches:")
        for item in mismatches[: args.show_mismatches]:
            location = item["file"]
            if "head" in item:
                location += f" {item['head']} {item['singular_vector']}"
            print(f"  {location}: {item['reason']}")
        if len(mismatches) > args.show_mismatches:
            remaining = len(mismatches) - args.show_mismatches
            print(f"  ... {remaining} more")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.json_output}")


if __name__ == "__main__":
    main()
