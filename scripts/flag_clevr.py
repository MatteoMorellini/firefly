#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

global_feature_counts = Counter()
global_attribute_stats = defaultdict(Counter)

FEATURES = {
    "Shape": [
        "cone", "cube", "cylinder", "diamond", "gear",
        "monkey", "sphere", "star", "teapot", "torus",
    ],
    "Color": [
        "blue", "brown", "cyan", "gray", "green",
        "orange", "pink", "purple", "red", "yellow",
    ],
    "Material": [
        "brick", "checkered", "chessboard", "circles", "emojis",
        "metal", "rubber", "star", "wave", "zigzag",
    ],
    "Count": [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "0th", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    ],
}

FEATURE_SETS = {
    feature: {concept.lower() for concept in concepts}
    for feature, concepts in FEATURES.items()
}

def collect_attribute_statistics(
    sv_entries: list[dict[str, Any]],
    count_duplicates: bool,
    excluded_features: set[str] | None = None,
) -> dict[str, Counter]:

    if excluded_features is None:
        excluded_features = set()

    stats = defaultdict(Counter)
    seen_per_feature = defaultdict(set)

    for item in sv_entries:
        if not isinstance(item, dict):
            continue

        concept = normalize_concept(item.get("concept"))
        if not concept:
            continue

        for feature, valid_concepts in FEATURE_SETS.items():
            if feature in excluded_features:
                continue

            if concept in valid_concepts:
                if count_duplicates:
                    stats[feature][concept] += 1
                else:
                    seen_per_feature[feature].add(concept)

    if not count_duplicates:
        for feature, concepts in seen_per_feature.items():
            for concept in concepts:
                stats[feature][concept] += 1

    return stats

def sv_sort_key(sv_name: str) -> int:
    """
    Convert 'sv-000' -> 0, 'sv-012' -> 12.
    Unknown formats are sent to the end.
    """
    try:
        return int(sv_name.split("-")[-1])
    except ValueError:
        return 10**9

def normalize_concept(concept: Any) -> str:
    return str(concept).strip().lower()


def iter_singular_vectors(data: dict[str, Any], first_sv_only: bool = False):
    """
    Expected structure:
    {
      "head-00": {
        "sv-000": [...],
        "sv-001": [...],
        ...
      },
      ...
    }

    If first_sv_only=True, only the first SV of each head is yielded.
    Usually this is sv-000.
    """
    for head_name, head_data in data.items():
        if not isinstance(head_data, dict):
            continue

        sv_items = [
            (sv_name, sv_entries)
            for sv_name, sv_entries in head_data.items()
            if isinstance(sv_entries, list)
        ]

        if not sv_items:
            continue

        sv_items = sorted(sv_items, key=lambda item: sv_sort_key(item[0]))

        if first_sv_only:
            sv_items = sv_items[:1]

        for sv_name, sv_entries in sv_items:
            yield head_name, sv_name, sv_entries


def analyze_sv(
    sv_entries: list[dict[str, Any]],
    threshold: int,
    count_duplicates: bool,
    excluded_features: set[str] | None = None,
) -> list[dict[str, Any]]:

    if excluded_features is None:
        excluded_features = set()

    feature_matches = defaultdict(list)

    for rank, item in enumerate(sv_entries):
        if not isinstance(item, dict):
            continue

        concept = normalize_concept(item.get("concept"))
        if not concept:
            continue

        for feature, valid_concepts in FEATURE_SETS.items():
            if feature in excluded_features:
                continue

            if concept in valid_concepts:
                feature_matches[feature].append(
                    {
                        "rank": rank,
                        "concept": concept,
                        "score": item.get("score"),
                        "index": item.get("index"),
                    }
                )

    flags = []

    for feature, matches in feature_matches.items():
        if count_duplicates:
            count = len(matches)
            concepts = [m["concept"] for m in matches]
        else:
            concepts = sorted({m["concept"] for m in matches})
            count = len(concepts)

        if count >= threshold:
            flags.append(
                {
                    "feature": feature,
                    "count": count,
                    "concepts": concepts,
                    "matches": matches,
                }
            )

    return flags


def collect_json_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "**/*.json" if recursive else "*.json"
    return sorted(input_path.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flag singular vectors where at least N concepts belong "
            "to the same feature category."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Folder containing JSON files, or a single JSON file.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Minimum number of concepts from the same feature required to flag an SV.",
    )

    parser.add_argument(
        "--exclude-colors",
        action="store_true",
        help="Exclude Color findings from the flagged results.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan JSON files directly inside the input folder.",
    )

    parser.add_argument(
        "--count-duplicates",
        action="store_true",
        help=(
            "Count repeated concepts multiple times. "
            "By default, each concept is counted once per SV."
        ),
    )
    parser.add_argument(
        "--first-sv-only",
        action="store_true",
        help="Only analyze the first singular vector of each head, usually sv-000.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for the JSON report.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path for the CSV report.",
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_base = input_path if input_path.is_dir() else input_path.parent

    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else output_base / "flagged_singular_vectors.json"
    )

    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else output_base / "flagged_singular_vectors.csv"
    )

    json_files = collect_json_files(
        input_path,
        recursive=not args.no_recursive,
    )

    results = []
    errors = []

    global_attribute_stats = defaultdict(Counter)

    for json_file in json_files:
        # Avoid reading previous reports if they are in the same folder.
        if json_file.resolve() in {output_json, output_csv}:
            continue

        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            errors.append(
                {
                    "file": str(json_file),
                    "error": str(exc),
                }
            )
            continue

        if not isinstance(data, dict):
            errors.append(
                {
                    "file": str(json_file),
                    "error": "Top-level JSON object is not a dictionary.",
                }
            )
            continue

        for head_name, sv_name, sv_entries in iter_singular_vectors(
            data,
            first_sv_only=args.first_sv_only,
        ):
            excluded_features = {"Color"} if args.exclude_colors else set()

            flags = analyze_sv(
                sv_entries,
                threshold=args.threshold,
                count_duplicates=args.count_duplicates,
                excluded_features=excluded_features,
            )

            sv_stats = collect_attribute_statistics(
                sv_entries,
                count_duplicates=args.count_duplicates,
                excluded_features=excluded_features,
            )

            for feature, counter in sv_stats.items():
                # attribute-level statistics
                global_attribute_stats[feature].update(counter)

                # category-level statistics
                global_feature_counts[feature] += sum(counter.values())

            for flag in flags:
                results.append(
                    {
                        "file": str(json_file.relative_to(output_base)),
                        "head": head_name,
                        "singular_vector": sv_name,
                        **flag,
                    }
                )

    report = {
        "threshold": args.threshold,
        "count_duplicates": args.count_duplicates,
        "first_sv_only": args.first_sv_only,
        "exclude_colors": args.exclude_colors,
        "excluded_features": sorted(excluded_features),
        "features": FEATURES,
        "num_files_scanned": len(json_files),
        "num_flagged_singular_vectors": len(results),

        # Whole-category counts
        "feature_statistics": {
            feature: count
            for feature, count in global_feature_counts.most_common()
        },

        # Attribute counts inside each category
        "attribute_statistics": {
            feature: dict(counter.most_common())
            for feature, counter in sorted(global_attribute_stats.items())
        },

        "results": results,
        "errors": errors,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "head",
                "singular_vector",
                "feature",
                "count",
                "concepts",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "file": row["file"],
                    "head": row["head"],
                    "singular_vector": row["singular_vector"],
                    "feature": row["feature"],
                    "count": row["count"],
                    "concepts": ", ".join(row["concepts"]),
                }
            )

    print(f"Scanned {len(json_files)} JSON file(s).")
    print(f"Flagged {len(results)} singular vector / feature match(es).")
    print(f"Wrote JSON report to: {output_json}")
    print(f"Wrote CSV report to:  {output_csv}")

    if errors:
        print(f"Warning: {len(errors)} file(s) had errors. See JSON report.")


if __name__ == "__main__":
    main()