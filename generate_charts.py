import sys
from math import pi
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:
    ttest_rel = None
    wilcoxon = None

# ── Constants ────────────────────────────────────────────────────────────────

METRICS = {
    "financial_accuracy": "Accuracy",
    "comprehensiveness": "Comprehensiveness",
    "diversity": "Diversity",
    "empowerment": "Empowerment",
    "directness": "Directness",
}

ARCHITECTURES = {
    "Vector RAG":  ("vector_rag_results.csv",  "#4A90E2"),
    "Graph RAG":   ("graph_rag_results.csv",   "#50E3C2"),
    "Hybrid RAG":  ("hybrid_rag_results.csv",  "#F5A623"),
}

SCORE_RANGE = (0, 5)
OUTPUT_DIR  = Path(".")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_scores(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV and return metric score columns with normalized names."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"{file_path}: missing columns {missing}")

    return df[list(METRICS)].copy()


def load_all() -> dict[str, pd.DataFrame]:
    """Load data for every architecture; exit with a helpful message on failure."""
    results = {}
    errors  = []

    for name, (path, _) in ARCHITECTURES.items():
        try:
            results[name] = load_scores(path)
        except FileNotFoundError:
            errors.append(f"  • {path!r} not found")
        except Exception as exc:
            errors.append(f"  • {path!r}: {exc}")

    if errors:
        print("Could not load all data files:\n" + "\n".join(errors))
        sys.exit(1)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def metric_values(data: dict[str, pd.DataFrame]) -> dict[str, list[float]]:
    """Return per-architecture mean metric scores in METRICS order."""
    return {
        name: [scores[m].mean() for m in METRICS]
        for name, scores in data.items()
    }


def metric_std_values(data: dict[str, pd.DataFrame]) -> dict[str, list[float]]:
    """Return per-architecture std-dev metric scores in METRICS order."""
    return {
        name: [scores[m].std(ddof=1) if len(scores) > 1 else 0.0 for m in METRICS]
        for name, scores in data.items()
    }


def add_bar_labels(ax: plt.Axes, rects, fmt: str = "{:.2f}", fontsize: int = 8) -> None:
    """Annotate each bar with its value."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize,
        )


def save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
    print(f"Saved → {svg_path}")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_bar_chart(values: dict[str, list[float]], std_values: dict[str, list[float]]) -> None:
    labels      = list(METRICS.values())
    names       = list(values.keys())
    colors      = [ARCHITECTURES[n][1] for n in names]
    n_groups    = len(labels)
    n_bars      = len(names)
    width       = 0.8 / n_bars                          # auto-scale bar width
    x           = np.arange(n_groups)
    offsets     = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, ax = plt.subplots(figsize=(11, 6))

    for offset, (name, vals), color in zip(offsets, values.items(), colors):
        rects = ax.bar(
            x + offset,
            vals,
            width,
            yerr=std_values[name],
            capsize=4,
            label=name,
            color=color,
            edgecolor="white",
            error_kw={"elinewidth": 1.2},
        )
        add_bar_labels(ax, rects)

    ax.set_ylabel("Average Score (1 – 5)", fontsize=12)
    ax.set_title("RAG Architecture Performance Across 5 Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(*SCORE_RANGE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    save(fig, OUTPUT_DIR / "comparison_barchart.png")


def plot_radar_chart(values: dict[str, list[float]]) -> None:
    labels  = list(METRICS.values())
    n       = len(labels)
    angles  = [i / n * 2 * pi for i in range(n)]
    angles += angles[:1]                                # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(pi / 2)                        # start at the top
    ax.set_theta_direction(-1)                         # clockwise

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="grey", fontsize=11)
    ax.set_rlabel_position(30)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="grey", fontsize=8)
    ax.set_ylim(*SCORE_RANGE)

    for name, vals in values.items():
        color       = ARCHITECTURES[name][1]
        radar_vals  = vals + [vals[0]]
        ax.plot(
            angles,
            radar_vals,
            linewidth=2,
            marker="o",
            markersize=5,
            label=name,
            color=color,
        )
        ax.fill(angles, radar_vals, color=color, alpha=0.15)

    ax.set_title("Architecture Shape Comparison", size=15, y=1.1, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    save(fig, OUTPUT_DIR / "comparison_radarchart.png")


def plot_box_chart(data: dict[str, pd.DataFrame]) -> None:
    records = []
    for name, scores in data.items():
        per_question_score = scores[list(METRICS)].mean(axis=1)
        records.extend(
            {"Architecture": name, "Score": score}
            for score in per_question_score
        )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))
    names = list(ARCHITECTURES.keys())
    grouped = [df.loc[df["Architecture"] == name, "Score"].to_numpy() for name in names]

    box = ax.boxplot(
        grouped,
        patch_artist=True,
        labels=names,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
    )

    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(ARCHITECTURES[name][1])
        patch.set_alpha(0.45)

    ax.set_title("Per-Question Score Distribution by Architecture", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score per Question (1 – 5)", fontsize=12)
    ax.set_ylim(*SCORE_RANGE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    save(fig, OUTPUT_DIR / "comparison_boxplot.png")


def run_significance_tests(data: dict[str, pd.DataFrame]) -> None:
    rows = []
    pairs = list(combinations(ARCHITECTURES.keys(), 2))

    for left, right in pairs:
        left_df = data[left]
        right_df = data[right]
        n = min(len(left_df), len(right_df))
        if n < 2:
            continue

        for metric in METRICS:
            x = left_df[metric].to_numpy()[:n]
            y = right_df[metric].to_numpy()[:n]

            if ttest_rel is not None:
                stat, pvalue = ttest_rel(x, y, nan_policy="omit")
                rows.append(
                    {
                        "comparison": f"{left} vs {right}",
                        "metric": metric,
                        "test": "paired_ttest",
                        "statistic": float(stat),
                        "p_value": float(pvalue),
                        "significant_0_05": bool(pvalue < 0.05),
                        "n": int(n),
                    }
                )

            if wilcoxon is not None:
                try:
                    w_stat, w_pvalue = wilcoxon(x, y, zero_method="wilcox")
                    rows.append(
                        {
                            "comparison": f"{left} vs {right}",
                            "metric": metric,
                            "test": "wilcoxon",
                            "statistic": float(w_stat),
                            "p_value": float(w_pvalue),
                            "significant_0_05": bool(w_pvalue < 0.05),
                            "n": int(n),
                        }
                    )
                except ValueError:
                    pass

    if not rows:
        print("Significance testing skipped (scipy missing, or not enough data).")
        return

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "significance_tests.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    data   = load_all()
    values = metric_values(data)
    std_values = metric_std_values(data)

    plot_bar_chart(values, std_values)
    plot_radar_chart(values)
    plot_box_chart(data)
    run_significance_tests(data)

    print("\nDone. Charts and statistical summary have been saved.")


if __name__ == "__main__":
    main()


