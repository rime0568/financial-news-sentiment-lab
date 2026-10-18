from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


INPUT_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")
OUTPUT_FILE = Path("outputs/tables/preliminary_regression_results_with_ticker_fe.csv")

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


def run_ols_with_ticker_fe(df, outcome, main_variable, controls):
    model_columns = [outcome, main_variable, "ticker"] + controls
    model_data = df[model_columns].dropna().copy()

    rhs_terms = [main_variable] + controls + ["C(ticker)"]
    formula = f"{outcome} ~ " + " + ".join(rhs_terms)

    model = smf.ols(formula=formula, data=model_data).fit()

    return {
        "formula": formula,
        "coefficient": model.params[main_variable],
        "standard_error": model.bse[main_variable],
        "t_stat": model.tvalues[main_variable],
        "p_value": model.pvalues[main_variable],
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
    }


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
# 2. Run preliminary OLS models with ticker fixed effects
# =========================

results = []

for spec in MODEL_SPECS:
    ols_result = run_ols_with_ticker_fe(
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
            "fixed_effects": "ticker",
            **ols_result,
        }
    )

results_df = pd.DataFrame(results)

# =========================
# 3. Save output
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 4. Print summary
# =========================

print("Input file checked:", INPUT_FILE)
print("Output saved to:", OUTPUT_FILE)
print("Required columns present:", True)
print("Rows in input sample:", len(sample))
print("Number of tickers:", sample["ticker"].nunique())
print("Ticker fixed effects included:", True)

print("\nPreliminary OLS results with ticker fixed effects:")
print(results_df.to_string(index=False))

print("\nImportant note:")
print("These are preliminary association models only. Ticker fixed effects control stable firm-level differences, but they do not establish causality.")
