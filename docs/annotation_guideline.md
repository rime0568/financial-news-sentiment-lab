# Human Sentiment Annotation Guideline

## Task

Assign one label to each document/headline based only on the text provided:

- `positive`: the text conveys a clearly favorable development for the company or investors relative to a reasonable baseline.
- `neutral`: the text is factual, descriptive, ambiguous, balanced, or does not convey a clear directional implication.
- `negative`: the text conveys a clearly unfavorable development, risk, deterioration, adverse legal/regulatory event, or below-expectation outcome.

## Rules

1. Do not use outside knowledge about the company or stock price.
2. Treat “beat” or “above expectations” as positive and “miss,” “below expectations,” “cut,” or “warning” as negative when the direction is explicit.
3. A positive result with weak guidance is `neutral` unless the adverse guidance clearly dominates the text; record the borderline case in `annotation_note`.
4. Legal, regulatory, or litigation language is negative only when the text describes an adverse event or material risk. A routine filing or investigation without direction is neutral.
5. If the text contains both favorable and unfavorable information and no side clearly dominates, use neutral.
6. If the text cannot be interpreted reliably, use neutral and explain why.

## Required fields

- `human_sentiment`: `positive`, `neutral`, or `negative`;
- `annotation_confidence`: `high`, `medium`, or `low`;
- `annotation_note`: short explanation for borderline cases.

The gold sample is a human validation set, not a model-generated label. Do not fill it with dictionary, ML, or LLM output.
