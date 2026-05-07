from __future__ import annotations

import argparse
import json
import os

from cluster.data.metadata import (
    build_canonical_metadata,
    build_metadata_features,
    save_metadata_feature_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical metadata features aligned to a processed music dataset."
    )
    parser.add_argument(
        "--aligned_root",
        type=str,
        required=True,
        help="Directory containing aligned_audio_metadata.csv and aligned_lyrics_metadata.csv.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Processed music dataset directory containing track_index.tsv.",
    )
    parser.add_argument(
        "--min_token_freq",
        type=int,
        default=3,
        help="Minimum token frequency to keep for list-like metadata fields.",
    )
    parser.add_argument(
        "--max_tokens_per_field",
        type=int,
        default=128,
        help="Maximum vocabulary size to keep for each list-like metadata field.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    canonical = build_canonical_metadata(
        aligned_root=str(args.aligned_root),
        processed_dir=str(args.processed_dir),
    )
    bundle = build_metadata_features(
        canonical_metadata=canonical,
        min_token_freq=int(args.min_token_freq),
        max_tokens_per_field=int(args.max_tokens_per_field),
    )
    written = save_metadata_feature_bundle(bundle, out_dir=str(args.processed_dir))

    summary = {
        "num_samples": int(bundle.features.shape[0]),
        "feature_dim": int(bundle.features.shape[1]),
        "min_token_freq": int(args.min_token_freq),
        "max_tokens_per_field": int(args.max_tokens_per_field),
        "written_files": written,
    }
    summary_path = os.path.join(str(args.processed_dir), "metadata_feature_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Metadata] Wrote metadata features to {args.processed_dir}")
    print(f"  - metadata.npy [{bundle.features.shape[0]}, {bundle.features.shape[1]}]")
    print("  - metadata_feature_names.json")
    print("  - metadata_vocab.json")
    print("  - canonical_metadata.csv")
    print("  - metadata_feature_summary.json")


if __name__ == "__main__":
    main()
