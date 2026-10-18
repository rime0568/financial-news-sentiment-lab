from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")
REFERENCE_FILE = Path("outputs/tables/preliminary_regression_results_with_ticker_fe.csv")
OUTPUT_FILE = Path("outputs/tables/preliminary_regression_results_with_ticker_fe_clustered_se.csv")

REQUIRED_COLUMNS = [
    "ticker",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "has_news",
    "log_news_count",
    "negative_headline_ratio",
    "daily_return",
]

MODEL_SPECS = [
    {
        "outcome": "next_day_abs_return",
        "model_name": "Model A + ticker FE",
        "main_variable": "has_news",
        "controls": ["daily_return"],
    },
    {
        "outcome": "next_day_abs_return",
        "model_name": "Model B + ticker FE",
        "main_variable": "log_news_count",
        "controls": ["daily_return"],
    },
    {
        "outcome": "next_day_abs_return",
        "model_name": "Model C + ticker FE",
        "main_variable": "negative_headline_ratio",
        "controls": ["daily_return"],
    },
    {
        "outcome": "next_day_abnormal_volume",
        "model_name": "Model A + ticker FE",
        "main_variable": "has_news",
        "controls": ["daily_return"],
    },
    {
        "outcome": "next_day_abnormal_volume",
        "model_name": "Model B + ticker FE",
        "main_variable": "log_news_count",
        "controls": ["daily_return"],
    },
    {
        "outcome": "next_day_abnormal_volume",
        "model_name": "Model C + ticker FE",
        "main_variable": "negative_headline_ratio",
        "controls": ["daily_return"],
    },
]


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")


def check_required_columns(df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


def run_ols_with_ticker_fe_clustered_se(df, outcome, main_variable, controls):
    model_columns = [outcome, main_variable, "ticker"] + controls
    model_data = df[model_columns].dropna().copy()

    rhs_terms = [main_variable] + controls + ["C(ticker)"]
    formula = f"{outcome} ~ " + " + ".join(rhs_terms)

    model = smf.ols(formula=formula, data=model_data).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data["ticker"]},
    )

    return {
        "formula": formula,
        "coefficient": model.params[main_variable],
        "standard_error_clustered": model.bse[main_variable],
        "t_stat_clustered": model.tvalues[main_variable],
        "p_value_clustered": model.pvalues[main_variable],
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
    }


def add_reference_comparison(results_df):
    if not REFERENCE_FILE.exists():
        print("\nReference file not found, skipping comparison:", REFERENCE_FILE)
        return results_df

    reference = pd.read_csv(REFERENCE_FILE)
    comparison_cols = [
        "outcome",
        "model_name",
        "main_variable",
        "coefficient",
        "standard_error",
        "p_value",
    ]
    missing_reference_cols = [col for col in comparison_cols if col not in reference.columns]
    if missing_reference_cols:
        print("\nReference file is missing comparison columns, skipping comparison.")
        return results_df

    reference = reference[comparison_cols].rename(
        columns={
            "coefficient": "coefficient_reference",
            "standard_error": "standard_error_reference",
            "p_value": "p_value_reference",
        }
    )

    merged = results_df.merge(
        reference,
        on=["outcome", "model_name", "main_variable"],
        how="left",
    )

    merged["coefficient_difference_vs_reference"] = (
        merged["coefficient"] - merged["coefficient_reference"]
    )
    merged["clustered_se_minus_reference_se"] = (
        merged["standard_error_clustered"] - merged["standard_error_reference"]
    )
    merged["clustered_se_larger_than_reference"] = (
        merged["standard_error_clustered"] > merged["standard_error_reference"]
    )
    merged["p_value_change_vs_reference"] = (
        merged["p_value_clustered"] - merged["p_value_reference"]
    )

    return merged


check_input_file()

# =========================
# 1. Load data and check columns
# =========================

sample = pd.read_csv(INPUT_FILE)
check_required_columns(sample)

sample["ticker"] = sample["ticker"].astype(str).str.upper().str.strip()

for col in REQUIRED_COLUMNS:
    if col != "ticker":
        sample[col] = pd.to_numeric(sample[col], errors="coerce")

# =========================
# 2. Run same ticker FE models with ticker-clustered standard errors
# =========================

results = []

for spec in MODEL_SPECS:
    ols_result = run_ols_with_ticker_fe_clustered_se(
        sample,
        outcome=spec["outcome"],
        main_variable=spec["main_variable"],
        controls=spec["controls"],
    )

    results.append(
        {
            "outcome": spec["outcome"],
            "model_name": spec["model_name"],
            "main_variable": spec["main_variable"],
            "controls": " + ".join(spec["controls"]),
            "has_ticker_fe": True,
            "clustered_by": "ticker",
            **ols_result,
        }
    )

results_df = pd.DataFrame(results)
results_df = add_reference_comparison(results_df)

# =========================
# 3. Save output
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 4. Print summary
# =========================

print("Input file checked:", INPUT_FILE)
print("Reference file checked:", REFERENCE_FILE.exists(), "-", REFERENCE_FILE)
print("Output saved to:", OUTPUT_FILE)
print("Required columns present:", True)
print("Rows in input sample:", len(sample))
print("Number of tickers:", sample["ticker"].nunique())
print("Ticker fixed effects included:", True)
print("Clustered standard errors by ticker:", True)

display_cols = [
    "outcome",
    "model_name",
    "main_variable",
    "coefficient",
    "standard_error_clustered",
    "t_stat_clustered",
    "p_value_clustered",
    "n_obs",
    "r_squared",
    "has_ticker_fe",
    "clustered_by",
]

print("\nTicker FE OLS results with ticker-clustered standard errors:")
print(results_df[display_cols].to_string(index=False))

comparison_cols = [
    "outcome",
    "model_name",
    "main_variable",
    "coefficient_difference_vs_reference",
    "standard_error_reference",
    "standard_error_clustered",
    "clustered_se_minus_reference_se",
    "p_value_reference",
    "p_value_clustered",
    "p_value_change_vs_reference",
]

if all(col in results_df.columns for col in comparison_cols):
    print("\nComparison with yesterday's ticker FE OLS:")
    print(results_df[comparison_cols].to_string(index=False))

print("\nImportant note:")
print("These are preliminary association models only. Clustered standard errors make inference more cautious for panel data, but they do not establish causality.")
