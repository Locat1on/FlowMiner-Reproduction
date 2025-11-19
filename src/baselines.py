import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def select_features(df: pd.DataFrame, mode: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"label", "direction"}]

    if mode == "length":
        keywords = ("packet_count", "byte_count", "pkt_len", "duration")
        return [c for c in numeric_cols if any(k in c for k in keywords)]
    if mode == "bytes":
        keywords = ("byte", "pkt_len")
        return [c for c in numeric_cols if any(k in c for k in keywords)]
    if mode == "full":
        return numeric_cols
    raise ValueError(f"Unknown feature set '{mode}'.")


def baseline_models(include_lightgbm: bool) -> Dict[str, object]:
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample"
        )
    }
    if include_lightgbm and LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            objective="multiclass",
            learning_rate=0.05,
            num_leaves=64,
            n_estimators=500,
        )
    return models


def run_baselines(
    nodes_csv: Path,
    feature_mode: str,
    folds: int,
    output_json: Optional[Path],
    include_lightgbm: bool,
) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(nodes_csv)
    feature_cols = select_features(df, feature_mode)
    X = df[feature_cols].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    model_names = list(baseline_models(include_lightgbm).keys())
    metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in model_names}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for name, model in baseline_models(include_lightgbm).items():
            clf = model
            if hasattr(clf, "random_state"):
                setattr(clf, "random_state", 42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, preds, average="macro", zero_division=0
            )
            metrics[name].append(
                {
                    "accuracy": accuracy_score(y_test, preds),
                    "precision": precision,
                    "recall": recall,
                    "f1_macro": f1,
                }
            )

    aggregated = {
        name: {
            "accuracy": float(np.mean([m["accuracy"] for m in runs])),
            "precision": float(np.mean([m["precision"] for m in runs])),
            "recall": float(np.mean([m["recall"] for m in runs])),
            "f1_macro": float(np.mean([m["f1_macro"] for m in runs])),
        }
        for name, runs in metrics.items()
        if runs
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_mode": feature_mode,
                    "folds": folds,
                    "metrics": aggregated,
                },
                f,
                indent=2,
            )
    return aggregated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classical baseline experiments for FlowMiner reproduction.")
    parser.add_argument("--nodes-csv", type=Path, default=Path("data/processed/nodes_enhanced.csv"))
    parser.add_argument("--feature-mode", choices=["length", "bytes", "full"], default="full")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=Path("results/reports/baselines.json"))
    parser.add_argument("--with-lightgbm", action="store_true", help="Include LightGBM baseline if installed.")
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    results = run_baselines(
        nodes_csv=parsed.nodes_csv,
        feature_mode=parsed.feature_mode,
        folds=parsed.folds,
        output_json=parsed.output_json,
        include_lightgbm=parsed.with_lightgbm,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
