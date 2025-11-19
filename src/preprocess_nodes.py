import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_dir, filter_small_classes


def preprocess_nodes(
    input_csv: Path,
    output_csv: Path,
    min_samples_per_class: int = 1,
    protected_numeric: Optional[Sequence[str]] = None,
) -> None:
    """Apply FlowMiner-style preprocessing on node features.

    Steps:
    1. Optionally filter out classes (labels) with too few samples.
    2. For all numeric columns (except protected ones), fill NaNs with column mean.
    3. Apply Min–Max scaling to [0, 1] per numeric column.
    """

    if protected_numeric is None:
        protected_numeric = ["label"]

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV {input_csv} does not exist")

    print(f"[Preprocess] Loading nodes from {input_csv} ...")
    df = pd.read_csv(input_csv)
    print(f"[Preprocess] Loaded {len(df)} rows with {len(df.columns)} columns.")
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in nodes CSV for class filtering.")

    # 1. Filter small classes if requested
    if min_samples_per_class != 1:
        # min_samples_per_class > 1: 使用显式阈值；
        # min_samples_per_class <= 0: 自动根据标签分布选择一个合理阈值。
        if min_samples_per_class > 1:
            threshold = min_samples_per_class
            mode = "explicit"
        else:
            counts = df["label"].value_counts()
            if len(counts) <= 2:
                threshold = 1
                mode = "auto_skip"
            else:
                median_count = float(counts.median())
                # 自动阈值：不小于 2，且约为“中位数的 10%”
                threshold = max(2, int(round(median_count * 0.1)))
                mode = "auto"
        if threshold > 1:
            kept_labels, dropped_labels = filter_small_classes(df["label"], threshold)
            before_rows = len(df)
            df = df[df["label"].isin(kept_labels)].reset_index(drop=True)
            print(
                f"[Preprocess] Filtered classes with <{threshold} samples "
                f"({mode} mode): {before_rows} -> {len(df)} rows. "
                f"Kept labels={sorted(kept_labels)}, dropped labels={sorted(dropped_labels)}"
            )

    # 2. Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    protected_set = set(protected_numeric)
    numeric_cols = [c for c in numeric_cols if c not in protected_set]

    print(f"[Preprocess] Normalizing {len(numeric_cols)} numeric columns ...")

    # 3. Mean imputation + Min–Max scaling
    for col in numeric_cols:
        col_series = df[col]
        mean_val = col_series.mean()
        df[col] = col_series.fillna(mean_val)

        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max <= col_min:
            # Constant or invalid column -> set to 0
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)

    print(f"[Preprocess] Saving preprocessed nodes to {output_csv} ...")
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    print("[Preprocess] Done.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess node features: filter small classes, fill missing values "
            "with mean, and apply Min–Max scaling."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/nodes_enhanced.csv"),
        help="Input nodes CSV (typically nodes_enhanced.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/nodes_preprocessed.csv"),
        help="Output CSV after preprocessing",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help=(
            "Minimum number of samples required per class; classes with fewer samples "
            "are removed. Set to >1 to mimic paper's removal of tiny classes."
        ),
    )
    return parser


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    preprocess_nodes(
        input_csv=parsed.input_csv,
        output_csv=parsed.output_csv,
        min_samples_per_class=parsed.min_samples_per_class,
    )


if __name__ == "__main__":
    main()
