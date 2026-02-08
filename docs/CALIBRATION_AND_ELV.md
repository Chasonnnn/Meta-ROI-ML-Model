# Calibration and ELV

This project converts predicted probabilities into money-like values:

`ELV = P(qualified_within_14d) * value_per_qualified`

If probabilities are not calibrated, ELV will be systematically too high or too low.

## What We Do

Training flow:
1. Train a base classifier (logreg or LightGBM)
2. Calibrate probabilities on a time-separated calibration split (default: sigmoid)
3. Evaluate on a future test split

## How To Read The Report

In `runs/<run_id>/report.html`:
- Calibration plot: predicted probability vs observed qualification rate
- ECE: a simple summary error (lower is better)
- Lift@k and capture@k: "if I contact the top k% of leads, what fraction of qualified leads do I capture?"

## Caveats

- Calibration depends on stable labeling rules and honest time splits.
- If joins are weak (low match rate), metrics are not trustworthy.

