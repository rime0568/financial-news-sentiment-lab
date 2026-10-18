# Saved Results

`metrics/` contains small aggregate evaluation tables. `figures/` contains selected non-identifying research figures.

The saved TF-IDF test result uses a 53-row, date-disjoint chronological holdout from 2020-05-14 through 2020-06-10:

- accuracy: 0.5660
- Macro-F1: 0.5108
- weighted F1: 0.5562

`tfidf_run_metadata.csv` records the validation-selected regularization value. `model_split_summary.csv` records the date range and class counts for every split.

No completed LLM result is included. Do not infer LLM performance from the request-preparation or output-validation scripts.
