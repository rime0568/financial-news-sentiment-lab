from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")
CORR_FILE = Path("outputs/tables/news_market_correlation_matrix.csv")
HAS_NEWS_FILE = Path("outputs/tables/has_news_market_reaction_comparison.csv")
BASELINE_REG_FILE = Path("outputs/tables/preliminary_regression_results.csv")
TICKER_FE_REG_FILE = Path("outputs/tables/preliminary_regression_results_with_ticker_fe.csv")
CLUSTERED_REG_FILE = Path("outputs/tables/preliminary_regression_results_with_ticker_fe_clustered_se.csv")
FIGURE_DIR = Path("outputs/figures")
FIGURE_INDEX_FILE = FIGURE_DIR / "FIGURE_INDEX.md"


REQUIRED_DATA_COLUMNS = [
    "news_count",
    "headline_sentiment_mean_filled",
    "negative_headline_ratio",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "large_downside_next_3day",
    "has_news",
]


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def format_axis(ax):
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_news_count_scatter(df):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(
        df["news_count"],
        df["next_day_abnormal_volume"],
        s=22,
        alpha=0.32,
        color="#2F6F9F",
        edgecolors="none",
    )
    ax.set_title("Exploratory Pilot Plot: News Count and Next-Day Abnormal Volume")
    ax.set_xlabel("news_count")
    ax.set_ylabel("next_day_abnormal_volume")
    ax.text(
        0.01,
        -0.18,
        "Note: Scatter plot for descriptive checking only; not causal evidence.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    format_axis(ax)
    save_figure(fig, FIGURE_DIR / "polished_news_count_vs_next_day_abnormal_volume.png")


def make_sentiment_scatter(df):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(
        df["headline_sentiment_mean_filled"],
        df["next_day_abs_return"],
        s=22,
        alpha=0.34,
        color="#3D7A4E",
        edgecolors="none",
    )
    ax.set_title("Exploratory Pilot Plot: Headline Sentiment and Next-Day Absolute Return")
    ax.set_xlabel("headline_sentiment_mean_filled")
    ax.set_ylabel("next_day_abs_return")
    ax.text(
        0.01,
        -0.20,
        "Note: No-news firm-days use filled value 0. Descriptive pilot plot only; not causal evidence.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    format_axis(ax)
    save_figure(fig, FIGURE_DIR / "polished_headline_sentiment_vs_next_day_abs_return.png")


def make_negative_ratio_downside(df):
    plot_df = df[["negative_headline_ratio", "large_downside_next_3day"]].dropna().copy()
    bins = [-0.001, 0, 0.25, 0.50, 1.0]
    labels = ["0", "(0, 0.25]", "(0.25, 0.50]", "(0.50, 1.00]"]
    plot_df["negative_ratio_bin"] = pd.cut(
        plot_df["negative_headline_ratio"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    summary = (
        plot_df.groupby("negative_ratio_bin", observed=False)
        .agg(
            large_downside_rate=("large_downside_next_3day", "mean"),
            n_obs=("large_downside_next_3day", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(
        data=summary,
        x="negative_ratio_bin",
        y="large_downside_rate",
        ax=ax,
        color="#7B5E8E",
    )
    for i, row in summary.iterrows():
        ax.text(
            i,
            row["large_downside_rate"] + 0.01,
            f"n={int(row['n_obs'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    ax.set_title("Exploratory Pilot Plot: Negative Headline Ratio and Large Downside Rate")
    ax.set_xlabel("negative_headline_ratio bin")
    ax.set_ylabel("Mean large_downside_next_3day")
    ax.set_ylim(0, max(0.05, summary["large_downside_rate"].max() * 1.25))
    ax.text(
        0.01,
        -0.22,
        "Note: Binned descriptive visualization for pilot sample; not causal evidence.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    format_axis(ax)
    save_figure(fig, FIGURE_DIR / "polished_negative_headline_ratio_vs_large_downside.png")


def make_has_news_group_comparison(df):
    group = (
        df.groupby("has_news")
        .agg(
            next_day_abs_return=("next_day_abs_return", "mean"),
            next_day_abnormal_volume=("next_day_abnormal_volume", "mean"),
            large_downside_next_3day=("large_downside_next_3day", "mean"),
        )
        .reset_index()
    )
    group["has_news"] = group["has_news"].astype(int).astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    metrics = [
        ("next_day_abs_return", "Next-Day Absolute Return"),
        ("large_downside_next_3day", "Large Downside Rate"),
        ("next_day_abnormal_volume", "Next-Day Abnormal Volume"),
    ]
    colors = ["#4C78A8", "#6B8E5A", "#B07C4F"]

    for ax, (col, title), color in zip(axes, metrics, colors):
        sns.barplot(data=group, x="has_news", y=col, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("has_news")
        ax.set_ylabel("Group mean")
        format_axis(ax)

    fig.suptitle("Exploratory Pilot Comparison: Market Reactions by News Availability", y=1.04)
    fig.text(
        0.01,
        -0.03,
        "Note: Group means in the 12-firm pilot sample; descriptive comparison only, not causal evidence.",
        fontsize=9,
        color="#555555",
    )
    save_figure(fig, FIGURE_DIR / "polished_has_news_group_comparison.png")


def normalize_model_name(name):
    return str(name).replace(" + ticker FE", "").strip()


def make_regression_coefficient_comparison():
    baseline = pd.read_csv(BASELINE_REG_FILE)
    ticker_fe = pd.read_csv(TICKER_FE_REG_FILE)
    clustered = pd.read_csv(CLUSTERED_REG_FILE)

    baseline = baseline.assign(model_group="Baseline OLS")
    ticker_fe = ticker_fe.assign(model_group="Ticker FE OLS")
    clustered = clustered.assign(model_group="Ticker FE + ticker-clustered SE")

    baseline["standard_error_plot"] = baseline["standard_error"]
    ticker_fe["standard_error_plot"] = ticker_fe["standard_error"]
    clustered["standard_error_plot"] = clustered["standard_error_clustered"]

    keep = ["outcome", "model_name", "main_variable", "coefficient", "standard_error_plot", "model_group"]
    combined = pd.concat([baseline[keep], ticker_fe[keep], clustered[keep]], ignore_index=True)
    combined["model_name_base"] = combined["model_name"].apply(normalize_model_name)

    focus = combined[
        (combined["outcome"] == "next_day_abs_return")
        & (combined["main_variable"].isin(["has_news", "log_news_count", "negative_headline_ratio"]))
    ].copy()
    focus["label"] = focus["main_variable"].map(
        {
            "has_news": "has_news",
            "log_news_count": "log_news_count",
            "negative_headline_ratio": "negative_headline_ratio",
        }
    )

    order = ["has_news", "log_news_count", "negative_headline_ratio"]
    groups = ["Baseline OLS", "Ticker FE OLS", "Ticker FE + ticker-clustered SE"]
    x = np.arange(len(order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, group_name in enumerate(groups):
        sub = focus[focus["model_group"] == group_name].set_index("label").reindex(order)
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            sub["coefficient"],
            width=width,
            label=group_name,
            yerr=sub["standard_error_plot"],
            capsize=3,
            alpha=0.88,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Coefficient on main news variable")
    ax.set_title("Exploratory Pilot Regression Comparison: Outcome = next_day_abs_return")
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.01,
        -0.25,
        "Note: Error bars use each specification's reported standard errors; exploratory association, not causal evidence.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    format_axis(ax)
    save_figure(fig, FIGURE_DIR / "polished_regression_coefficient_comparison.png")


def make_correlation_heatmap():
    corr = pd.read_csv(CORR_FILE, index_col=0)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Exploratory Pilot Heatmap: News Variables and Market Reactions")
    ax.set_xlabel("Market reaction variables")
    ax.set_ylabel("News variables")
    fig.text(
        0.01,
        -0.02,
        "Note: Correlations are descriptive checks in the 12-firm pilot sample; not causal evidence.",
        fontsize=9,
        color="#555555",
    )
    save_figure(fig, FIGURE_DIR / "polished_correlation_heatmap.png")


def make_figure_index():
    text = """# Figure Index

This index documents the polished figures created for the RA interview research summary. All figures are exploratory pilot visualizations and should not be interpreted as causal evidence.

| Figure | Data files used | What it shows | Suggested summary section | Interpretation notes | RA interview use |
|---|---|---|---|---|---|
| `polished_news_count_vs_next_day_abnormal_volume.png` | `data/processed/analysis_sample_12firms_2020H1.csv` | Scatter plot of `news_count` vs. `next_day_abnormal_volume`. | Empirical Workflow / Discussion | Descriptive check only; abnormal volume is noisy and sensitive to clustered SE. | Optional |
| `polished_headline_sentiment_vs_next_day_abs_return.png` | `data/processed/analysis_sample_12firms_2020H1.csv` | Scatter plot of `headline_sentiment_mean_filled` vs. `next_day_abs_return`. | Variable Construction / Discussion | No-news firm-days use filled value 0; exploratory only. | Optional |
| `polished_negative_headline_ratio_vs_large_downside.png` | `data/processed/analysis_sample_12firms_2020H1.csv` | Binned descriptive plot of `negative_headline_ratio` and mean `large_downside_next_3day`. | Preliminary Findings / Future Extensions | Descriptive visualization for binary downside outcome; not causal evidence. | Yes, if discussing downside risk carefully |
| `polished_has_news_group_comparison.png` | `data/processed/analysis_sample_12firms_2020H1.csv`; `outputs/tables/has_news_market_reaction_comparison.csv` | Group means of selected market reactions by `has_news`. | Empirical Workflow / Preliminary Findings | Group mean comparison only; does not control for firm/date differences. | Yes |
| `polished_regression_coefficient_comparison.png` | `outputs/tables/preliminary_regression_results.csv`; `outputs/tables/preliminary_regression_results_with_ticker_fe.csv`; `outputs/tables/preliminary_regression_results_with_ticker_fe_clustered_se.csv` | Coefficient comparison across baseline OLS, ticker FE, and ticker-clustered SE for `next_day_abs_return`. | Preliminary Findings | Shows coefficient direction and uncertainty; not causal evidence. | Yes |
| `polished_correlation_heatmap.png` | `outputs/tables/news_market_correlation_matrix.csv` | Correlations between news variables and market reaction variables. | Empirical Workflow / Preliminary Findings | Correlation is descriptive and does not imply causality. | Yes |

PNG and PDF versions were saved for each polished figure.
"""
    FIGURE_INDEX_FILE.write_text(text, encoding="utf-8")


def main():
    for path in [
        DATA_FILE,
        CORR_FILE,
        HAS_NEWS_FILE,
        BASELINE_REG_FILE,
        TICKER_FE_REG_FILE,
        CLUSTERED_REG_FILE,
    ]:
        check_file(path)

    df = pd.read_csv(DATA_FILE)
    missing = [col for col in REQUIRED_DATA_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in analysis sample: {missing}")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    make_news_count_scatter(df)
    make_sentiment_scatter(df)
    make_negative_ratio_downside(df)
    make_has_news_group_comparison(df)
    make_regression_coefficient_comparison()
    make_correlation_heatmap()
    make_figure_index()

    print("Saved polished figures to:", FIGURE_DIR)
    print("Saved figure index to:", FIGURE_INDEX_FILE)


if __name__ == "__main__":
    main()
