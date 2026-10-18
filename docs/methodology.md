# Methodology and Closure Rules

## Research question

Can transparent and modern financial-text measures provide different information about short-term market reactions, and how do dictionary, traditional machine-learning and structured LLM labels compare against a manually reviewed sample?

## Data flow

FNSPID is the primary historical source. A small, compliant web-collected supplement can be used to demonstrate a second public source. Sources share a schema but remain identifiable through the `source` field.

Web records are not silently appended to the historical regression sample. Records require reliable company, publication-date and market-date matching before entering a separately identified analysis sample.

## Text and label sequence

1. Preserve source records and retrieval metadata locally.
2. Clean text, normalise company identifiers, parse dates and deduplicate using source identifiers and content hashes.
3. Build a unified document table with explicit provenance.
4. Draw a stratified human-reviewed sample with positive, neutral and negative labels.
5. Evaluate dictionary, TF-IDF logistic-regression and configured LLM labels against the same sample.
6. Freeze the model-selection rule before inspecting market regressions.
7. Aggregate the selected measure to firm-trading-day level.

## Timing rule

The conservative primary rule maps a document to the next trading day. Weekend and holiday documents roll forward. Missing or unreliable publication times remain flagged rather than guessed.

## Market analysis

Primary outcomes are next-day absolute return and abnormal volume. OLS, ticker fixed effects and ticker-clustered standard errors are baseline comparisons. Results are described as exploratory associations, not causal effects, crash predictions or trading performance.

## Leakage and source controls

- Do not split near-duplicate or same-event documents randomly across train and test sets.
- Do not use future market outcomes in text labels or prompts.
- Do not double-count the same event across sources.
- Preserve source-specific counts and results.
- Report model selection using the reviewed text sample rather than market significance.
