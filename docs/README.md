# Workflow Reference

The project separates data provenance, human review, model evaluation and market analysis so that a model result cannot silently become a financial claim.

```text
source records
  -> unified schema and content hashes
  -> human-reviewed sentiment sample
  -> dictionary / TF-IDF / structured LLM labels
  -> common evaluation
  -> frozen measure selection
  -> firm-day aggregation
  -> exploratory market-reaction analysis
```

The numbered script folders implement the first six stages. The root-level analysis scripts retain the original firm-day and regression workflow.

## Closure rule

A model may be reported only after its labels are compared against the same reviewed sample and the run metadata, predictions and errors are saved. The existence of a prompt or API script is not treated as a completed model result.
