import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# 尝试使用 seaborn 样式，如果不可用则手动设置
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Fallback for older matplotlib versions or if style not found
    plt.style.use("ggplot")

# 全局美化：更柔和的配色、稍大的字号和线宽
mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.figsize": (12, 8),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Microsoft YaHei", "SimHei"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# 自定义配色方案 (Colorblind friendly)
COLORS = {
    "train_loss": "#1f77b4",  # Blue
    "val_loss": "#ff7f0e",    # Orange
    "val_acc": "#2ca02c",     # Green
    "val_prec": "#d62728",    # Red
    "val_rec": "#9467bd",     # Purple
    "val_f1": "#8c564b",      # Brown
}

def smooth_curve(values: List[float], weight: float = 0.6) -> List[float]:
    """
    使用指数移动平均 (EMA) 平滑曲线
    """
    if not values:
        return []
    
    # 找到第一个非 None 值作为初始值
    last = None
    for v in values:
        if v is not None:
            last = v
            break
            
    if last is None: # 全是 None
        return values

    smoothed = []
    for v in values:
        if v is None:
            smoothed.append(None)
            continue
        
        # EMA 公式: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        # 这里 weight 对应 (1-alpha)，即历史权重的保留比例
        # 通常 weight=0.6 表示保留 60% 的历史，40% 的新值
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_history(log_path: Path) -> Dict:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file {log_path} does not exist")
    data = json.loads(log_path.read_text())
    history: List[Dict] = data.get("history", [])
    if not history:
        raise ValueError(f"Log file {log_path} does not contain 'history' entries")
    return {"meta": data, "history": history}


def plot_curves(history: Sequence[Dict], output_path: Path, title: str, shade_std: bool) -> None:
    epochs = [int(entry["epoch"]) for entry in history]

    def extract_series(key: str) -> Dict[str, List[float]] | None:
        values = [entry.get(key) for entry in history]
        if not values or all(v is None for v in values):
            return None
        std_values = [entry.get(f"{key}_std") for entry in history]
        return {"values": values, "std": std_values}

    # 使用更宽屏比例，方便看曲线细节
    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, y=0.96, fontsize=18, fontweight="bold")

    def plot_with_band(
        ax, 
        series: Dict[str, List[float]], 
        label: str, 
        color: str, 
        style: str = "-", 
        smooth: bool = True
    ):
        values = series["values"]
        std_values = series["std"]
        
        # 绘制原始数据的浅色背景线 (如果启用平滑)
        if smooth:
            ax.plot(
                epochs,
                values,
                style,
                color=color,
                alpha=0.25,
                linewidth=1.0,
                label=None  # 不在图例中显示原始线
            )
            # 计算平滑曲线
            plot_values = smooth_curve([v if v is not None else 0.0 for v in values], weight=0.7)
        else:
            plot_values = values

        # 绘制主曲线
        (line,) = ax.plot(
            epochs,
            plot_values,
            style,
            label=label,
            color=color,
            marker=None, # 平滑曲线通常不加 marker，除非点很少
            linewidth=2.5,
        )
        
        # 绘制标准差阴影
        if shade_std and std_values and any(std is not None for std in std_values):
            # 注意：这里用原始值加减标准差，而不是平滑值，或者也可以平滑上下界
            # 为了视觉效果，这里使用原始值的上下界
            lower = [v - std if std is not None and v is not None else v for v, std in zip(values, std_values)]
            upper = [v + std if std is not None and v is not None else v for v, std in zip(values, std_values)]
            ax.fill_between(epochs, lower, upper, alpha=0.15, color=color, edgecolor=None)
        
        return color

    def annotate_best(ax, series: Dict[str, List[float]], label: str, color: str, text_offset: Tuple[int, int]) -> None:
        values = series["values"]
        candidates = [
            (epoch, value)
            for epoch, value in zip(epochs, values)
            if value is not None
        ]
        if not candidates:
            return
        best_epoch, best_value = max(candidates, key=lambda item: item[1])
        
        # 在最高点画个圈
        ax.scatter(
            best_epoch,
            best_value,
            color=color,
            edgecolors="white",
            linewidths=1.5,
            s=80,
            zorder=10,
        )
        # 添加注释文本
        ax.annotate(
            f"{label}: {best_value:.4f}\n(Ep {best_epoch})",
            xy=(best_epoch, best_value),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color, alpha=0.6),
        )

    # --- Plot Loss ---
    train_loss = extract_series("train_loss")
    val_loss = extract_series("val_loss")
    
    if train_loss:
        plot_with_band(ax_loss, train_loss, "Train Loss", COLORS["train_loss"], style="-")
    if val_loss:
        plot_with_band(ax_loss, val_loss, "Val Loss", COLORS["val_loss"], style="--")
    
    ax_loss.set_ylabel("Loss", fontweight="bold")
    ax_loss.legend(frameon=True, framealpha=0.95, loc="upper right", shadow=True)
    ax_loss.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))

    # --- Plot Metrics ---
    val_acc = extract_series("val_accuracy")
    val_prec = extract_series("val_precision")
    val_rec = extract_series("val_recall")
    val_f1 = extract_series("val_f1")

    if val_acc:
        c = COLORS["val_acc"]
        plot_with_band(ax_metric, val_acc, "Accuracy", c, style="-")
        annotate_best(ax_metric, val_acc, "Acc", c, (20, 10))
    if val_prec:
        c = COLORS["val_prec"]
        plot_with_band(ax_metric, val_prec, "Precision", c, style="--")
        # annotate_best(ax_metric, val_prec, "Prec", c, (20, -20)) # 避免注释过多拥挤
    if val_rec:
        c = COLORS["val_rec"]
        plot_with_band(ax_metric, val_rec, "Recall", c, style=":")
        # annotate_best(ax_metric, val_rec, "Rec", c, (20, -40))
    if val_f1:
        c = COLORS["val_f1"]
        plot_with_band(ax_metric, val_f1, "F1-Score", c, style="-.")
        annotate_best(ax_metric, val_f1, "F1", c, (20, -30))

    ax_metric.set_xlabel("Epoch", fontweight="bold")
    ax_metric.set_ylabel("Score", fontweight="bold")
    
    # 自动调整 Y 轴范围，但保持一定的余量
    metric_values = []
    for s in [val_acc, val_prec, val_rec, val_f1]:
        if s:
            metric_values.extend(v for v in s["values"] if v is not None)
    
    if metric_values:
        min_val, max_val = min(metric_values), max(metric_values)
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.05
        ax_metric.set_ylim(max(0, min_val - margin), min(1.02, max_val + margin))
    else:
        ax_metric.set_ylim(0, 1.02)

    ax_metric.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_metric.legend(frameon=True, framealpha=0.95, loc="lower right", shadow=True, ncol=2)
    
    for axis in (ax_loss, ax_metric):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        # 移除上方和右侧的边框线，更美观
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight') # 高 DPI 保存
    plt.close(fig)


def aggregate_histories(histories: Sequence[Sequence[Dict]]) -> List[Dict]:
    metric_keys = [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
    ]
    aggregated: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {key: [] for key in metric_keys})
    for history in histories:
        for entry in history:
            epoch = entry["epoch"]
            for key in metric_keys:
                value = entry.get(key)
                if value is not None:
                    aggregated[epoch][key].append(value)

    summary: List[Dict] = []
    for epoch in sorted(aggregated.keys()):
        bucket = aggregated[epoch]
        record: Dict[str, float | None] = {"epoch": epoch}
        for key, values in bucket.items():
            if values:
                record[key] = fmean(values)
                record[f"{key}_std"] = stdev(values) if len(values) >= 2 else None
            else:
                record[key] = None
                record[f"{key}_std"] = None
        summary.append(record)
    return summary


def resolve_log_paths(log_paths: List[Path] | None, log_dir: Path | None) -> List[Path]:
    collected: List[Path] = []
    if log_paths:
        collected.extend(log_paths)
    if log_dir:
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory {log_dir} does not exist")
        if not log_dir.is_dir():
            raise NotADirectoryError(f"{log_dir} is not a directory")
        collected.extend(sorted(log_dir.glob("run_*_log.json")))
    if not collected:
        raise ValueError("No log files provided. Use --log or --log-dir.")

    resolved: List[Path] = []
    seen = set()
    for path in collected:
        p = path.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Log file {p} does not exist")
        if p not in seen:
            seen.add(p)
            resolved.append(p)
    return resolved


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FlowMiner training curves from a run log JSON.")
    parser.add_argument(
        "--log",
        dest="logs",
        type=Path,
        nargs="+",
        default=None,
        metavar="LOG",
        help="One or more run_*_log.json files to plot. If multiple, curves are averaged.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory containing run_*_log.json files (e.g., results/logs). All matching files will be used.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save the generated plot (PNG). Defaults next to log."
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    log_paths = resolve_log_paths(args.logs, args.log_dir)
    histories = [load_history(path)["history"] for path in log_paths]

    if len(histories) == 1:
        history_to_plot = histories[0]
        title = f"Training curves - {log_paths[0].stem}"
    else:
        history_to_plot = aggregate_histories(histories)
        title = f"Training curves - {len(histories)}-run average"

    output = args.output
    if output is None:
        if len(log_paths) == 1:
            output = log_paths[0].with_suffix(".png")
        else:
            default_dir = log_paths[0].parent
            output = default_dir / "runs_average.png"

    plot_curves(history_to_plot, output, title, shade_std=len(histories) > 1)
    print(f"[Plot] Saved figure to {output}")


if __name__ == "__main__":
    main()
