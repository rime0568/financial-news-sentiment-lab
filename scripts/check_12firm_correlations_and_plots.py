from pathlib import Path
import struct
import zlib

import pandas as pd


INPUT_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

CORRELATION_OUTPUT = TABLE_DIR / "news_market_correlation_matrix.csv"
HAS_NEWS_COMPARISON_OUTPUT = TABLE_DIR / "has_news_market_reaction_comparison.csv"

FIGURE_NEWS_COUNT_VOLUME = FIGURE_DIR / "news_count_vs_next_day_abnormal_volume.png"
FIGURE_SENTIMENT_ABS_RETURN = FIGURE_DIR / "headline_sentiment_vs_next_day_abs_return.png"
FIGURE_NEGATIVE_RATIO_DOWNSIDE = FIGURE_DIR / "negative_headline_ratio_vs_large_downside.png"

NEWS_VARIABLES = [
    "has_news",
    "news_count",
    "log_news_count",
    "headline_sentiment_mean_filled",
    "positive_headline_count",
    "negative_headline_count",
    "positive_headline_ratio",
    "negative_headline_ratio",
]

MARKET_REACTION_VARIABLES = [
    "next_day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "downside_next_day",
    "large_downside_next_3day",
]

REQUIRED_COLUMNS = ["ticker", "date"] + NEWS_VARIABLES + MARKET_REACTION_VARIABLES


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")


def check_required_columns(df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


def write_png(path, width, height, pixels):
    raw_rows = []
    for y in range(height):
        row = bytearray()
        row.append(0)
        for x in range(width):
            row.extend(pixels[y][x])
        raw_rows.append(bytes(row))

    raw_data = b"".join(raw_rows)

    def chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_data))
        + chunk(b"IEND", b"")
    )

    with open(path, "wb") as file:
        file.write(png)


def draw_point(pixels, x, y, color, radius=2):
    height = len(pixels)
    width = len(pixels[0])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                px = x + dx
                py = y + dy
                if 0 <= px < width and 0 <= py < height:
                    pixels[py][px] = color


def draw_line(pixels, x1, y1, x2, y2, color):
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    for step in range(steps + 1):
        x = int(round(x1 + (x2 - x1) * step / steps))
        y = int(round(y1 + (y2 - y1) * step / steps))
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
            pixels[y][x] = color


def save_scatter_plot(df, x_col, y_col, output_file, title, x_label, y_label):
    plot_df = df[[x_col, y_col]].dropna().copy()
    width, height = 900, 600
    left, right, top, bottom = 90, 40, 55, 80
    white = (255, 255, 255)
    axis = (60, 60, 60)
    grid = (220, 220, 220)
    point = (35, 105, 180)
    pixels = [[white for _ in range(width)] for _ in range(height)]

    x_min = float(plot_df[x_col].min())
    x_max = float(plot_df[x_col].max())
    y_min = float(plot_df[y_col].min())
    y_max = float(plot_df[y_col].max())

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    for i in range(6):
        x = int(plot_left + (plot_right - plot_left) * i / 5)
        y = int(plot_bottom - (plot_bottom - plot_top) * i / 5)
        draw_line(pixels, x, plot_top, x, plot_bottom, grid)
        draw_line(pixels, plot_left, y, plot_right, y, grid)

    draw_line(pixels, plot_left, plot_bottom, plot_right, plot_bottom, axis)
    draw_line(pixels, plot_left, plot_top, plot_left, plot_bottom, axis)

    for _, row in plot_df.iterrows():
        x_value = float(row[x_col])
        y_value = float(row[y_col])
        x_pixel = int(plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y_pixel = int(plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        draw_point(pixels, x_pixel, y_pixel, point, radius=2)

    # A small companion text file records labels because this minimal PNG writer does not draw fonts.
    label_file = output_file.with_suffix(".txt")
    label_file.write_text(
        "\n".join(
            [
                title,
                f"x-axis: {x_label}",
                f"y-axis: {y_label}",
                f"x range: {x_min:.6g} to {x_max:.6g}",
                f"y range: {y_min:.6g} to {y_max:.6g}",
                f"points plotted: {len(plot_df)}",
            ]
        ),
        encoding="utf-8",
    )

    write_png(output_file, width, height, pixels)


check_input_file()

# =========================
# 1. Load data and check columns
# =========================

sample = pd.read_csv(INPUT_FILE)
check_required_columns(sample)

sample["ticker"] = sample["ticker"].astype(str).str.upper().str.strip()
sample["date"] = pd.to_datetime(sample["date"], errors="coerce")

for col in NEWS_VARIABLES + MARKET_REACTION_VARIABLES:
    sample[col] = pd.to_numeric(sample[col], errors="coerce")

# =========================
# 2. Correlation matrix
# =========================

correlation_matrix = sample[NEWS_VARIABLES + MARKET_REACTION_VARIABLES].corr(method="pearson")
news_market_correlations = correlation_matrix.loc[NEWS_VARIABLES, MARKET_REACTION_VARIABLES]

# =========================
# 3. Mean comparison by has_news
# =========================

has_news_comparison = (
    sample.groupby("has_news")[MARKET_REACTION_VARIABLES]
    .agg(["count", "mean", "std", "median"])
)
has_news_comparison.columns = [
    f"{variable}_{stat}" for variable, stat in has_news_comparison.columns
]
has_news_comparison = has_news_comparison.reset_index()

# =========================
# 4. Save tables
# =========================

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

news_market_correlations.to_csv(CORRELATION_OUTPUT, encoding="utf-8-sig")
has_news_comparison.to_csv(HAS_NEWS_COMPARISON_OUTPUT, index=False, encoding="utf-8-sig")

# =========================
# 5. Save exploratory plots
# =========================

save_scatter_plot(
    sample,
    "news_count",
    "next_day_abnormal_volume",
    FIGURE_NEWS_COUNT_VOLUME,
    "News Count vs Next-Day Abnormal Volume",
    "news_count",
    "next_day_abnormal_volume",
)

save_scatter_plot(
    sample,
    "headline_sentiment_mean_filled",
    "next_day_abs_return",
    FIGURE_SENTIMENT_ABS_RETURN,
    "Headline Sentiment vs Next-Day Absolute Return",
    "headline_sentiment_mean_filled",
    "next_day_abs_return",
)

save_scatter_plot(
    sample,
    "negative_headline_ratio",
    "large_downside_next_3day",
    FIGURE_NEGATIVE_RATIO_DOWNSIDE,
    "Negative Headline Ratio vs Large Downside Indicator",
    "negative_headline_ratio",
    "large_downside_next_3day",
)

# =========================
# 6. Print readable diagnostics
# =========================

print("Input file checked:", INPUT_FILE)
print("Required columns present:", True)
print("Rows:", len(sample))
print("Firms:", sample["ticker"].nunique())
print("Date range:", sample["date"].min().strftime("%Y-%m-%d"), "to", sample["date"].max().strftime("%Y-%m-%d"))

print("\nNews variables:")
print(NEWS_VARIABLES)

print("\nMarket reaction variables:")
print(MARKET_REACTION_VARIABLES)

print("\nNews-market correlation matrix:")
print(news_market_correlations)

print("\nMarket reaction comparison by has_news:")
print(has_news_comparison.to_string(index=False))

print("\nSaved tables:")
print(CORRELATION_OUTPUT)
print(HAS_NEWS_COMPARISON_OUTPUT)

print("\nSaved figures:")
print(FIGURE_NEWS_COUNT_VOLUME)
print(FIGURE_SENTIMENT_ABS_RETURN)
print(FIGURE_NEGATIVE_RATIO_DOWNSIDE)
