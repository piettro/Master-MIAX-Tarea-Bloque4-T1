# Trustworthy Neural Networks for Credit Scoring: Fairness & Uncertainty

> Master's in AI & Quantum Computing Applied to Financial Markets (MIAX)
> Course: Advanced Artificial Intelligence (Block 4) | Assignment: B4-T1 Workshop

## Overview

A neural credit-default classifier for the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset. It is designed to be **accurate**, **fair** with respect to gender,
and **honest about its confidence**. The project covers the four tasks of
the brief:

1. **Custom architecture**: `DebtRatioLayer` computes two monetary ratios
   inside the network (annuity/income and credit/income). It reshapes each
   ratio with a trainable exponent clipped to `[0.1, 3]` and saturates it
   with `tanh(r^a / cap)` before the dense layers.
2. **Fair learning**: `FairLoss = weighted BCE + λ · corr(ŷ, gender)²`. The
   Pearson correlation is computed with differentiable `keras.ops`, and
   gender is passed to the loss as a second label column.
3. **AutoML**: Keras Tuner searches depth, width, activation, dropout,
   learning rate and λ, using the objective `val_AUC − |corr|`.
4. **Uncertainty**: each test prediction returns its class together with
   (a) the expected absolute error from an auxiliary error-predictor network
   (the approach shown in class) and (b) the MC-dropout predictive variance.

## Project Structure

```
├── main.py                     # single entry point (CLI)
├── src/
│   ├── pipeline.py             # end-to-end orchestration
│   ├── data/                   # loader.py, preprocessing.py
│   ├── models/                 # layers.py, losses.py, metrics.py,
│   │                           # classifier.py, uncertainty.py
│   ├── training/               # trainer.py, tuning.py (Keras Tuner)
│   ├── evaluation/             # metrics.py, selection.py, reporting.py,
│   │                           # uncertainty_analysis.py
│   ├── visualization/          # plots.py
│   └── utils/                  # config.py, logging_config.py,
│                               # reproducibility.py
├── tests/                      # 37 pytest tests (unit + end-to-end)
├── notebooks/walkthrough.ipynb # exploration of the generated outputs
├── outputs/                    # figures/, tables/, models/, run_summary.json
├── data/application_train.zip  # dataset (read directly, no unzip needed)
├── docs/architecture.md        # Mermaid diagrams
├── materials/                  # brief, lecture transcripts (reference)
├── scripts/setup_env.ps1       # optional one-shot Windows setup
├── requirements.txt
└── setup.cfg                   # flake8 + pytest configuration
```

## Setup & Installation

Requires **Python 3.10**.

```bash
# 1. Clone the repo
git clone <repo-url>
cd miax-b4t1

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

The dataset ships as `data/application_train.zip` and is read directly from
the zip. If you have extracted `data/application_train.csv`, the CSV is used
instead.

## Usage

```bash
python main.py            # full experiment: ~20 min on a 16-core CPU
python main.py --quick    # smoke test on a 6 000-row sample (~2 min)
python -m pytest          # run the test suite
```

| Option | Description |
|---|---|
| `--quick` | Small sample, 3 epochs, 2 tuner trials |
| `--output-dir PATH` | Where figures/tables/models are written (default `outputs/`) |
| `--data-path PATH` | Alternative `application_train.csv` or `.zip` |
| `--max-trials N` | Keras Tuner budget (default 20) |
| `--seed N` | Global random seed (default 42) |
| `--log-level LEVEL` | `DEBUG`, `INFO`, ... |

All hyper-parameters live in `src/utils/config.py`.

## Methodology

- **Data**: 307,507 applicants (4 with unknown gender are dropped). The
  data is split 80/10/10 with stratification. Monetary amounts are
  `log1p`-transformed, and every `EXT_SOURCE` gets a missing-value flag.
  Median imputation and standardisation are **fitted on the training split
  only**.
- **Selection protocol**: every candidate (8-value λ sweep plus 20 tuner
  trials) is scored on **validation**. The best FAIR model is the one with
  the highest validation ROC-AUC among those with validation
  |corr(ŷ, gender)| ≤ 0.05. Base (λ = 0) and FAIR are then retrained with
  the **same topology**, and the **test set is evaluated once**.
- **Class imbalance** (8.1% defaults) is handled with balanced class
  weights inside the loss.

## Results

Test set (30,751 applicants). The FAIR row is the selected model; the best
value in each column is in bold. The selected model is λ = 0.5 with a
64-32 ReLU topology, dropout 0.2.

| Model | Accuracy | Balanced acc. | ROC-AUC | PR-AUC | \|Pearson(ŷ,s)\| | \|Spearman(ŷ,s)\| | Demographic parity diff. | Equal opportunity diff. |
|---|---|---|---|---|---|---|---|---|
| Base (λ=0) | 0.6815 | **0.6766** | **0.7391** | **0.2186** | 0.2495 | 0.2502 | 0.2052 | 0.1734 |
| **Best FAIR (λ=0.5)** | **0.6857** | 0.6756 | 0.7348 | 0.2165 | **0.0322** | **0.0297** | **0.0492** | **0.0257** |

**Cost of fairness:** the FAIR loss removes 87% of the dependence on gender
(|corr| drops from 0.250 to 0.032). The demographic-parity gap falls from
20.5 to 4.9 points. The cost is **0.4 points of ROC-AUC**. On validation
the Pareto front is very steep: λ = 2 already reaches |corr| = 0.004 for
0.7 AUC points. Beyond λ ≈ 10, AUC drops quickly without any extra
fairness gain (λ = 50 gives AUC 0.718).

**Uncertainty vs EXT_SOURCE quality:** yes, the model is less certain when
the external scores are missing. The expected error rises monotonically with
the number of imputed `EXT_SOURCE` values:

| Missing EXT_SOURCE | n | Expected \|y − p\| | MC-dropout variance |
|---|---|---|---|
| 0 | 10,949 | 0.393 | 0.00140 |
| 1 | 16,130 | 0.424 | 0.00128 |
| 2 | 3,654 | 0.493 | 0.00128 |
| 3 | 18 | 0.537 | 0.00180 |

**Good vs bad payers:** predicted bad payers have a much higher expected
error (0.58 vs 0.34). With 8% defaults and balanced weights, most flagged
applicants sit close to p ≈ 0.5–0.7, so the decision is genuinely
uncertain. The MC-dropout variance is small (~1e-3) and similar across
groups. With ~246k training rows, the model's *epistemic* uncertainty is
low. The uncertainty that matters here is *aleatoric*: it lives in the data
(missing scores, overlapping classes), and the error model captures it. On
test, the error model's predictions correlate strongly with the realised
errors (Pearson 0.84, `figures/error_model_calibration.png`).

Generated deliverables (`outputs/`):

| File | Content |
|---|---|
| `figures/pareto_fairness.png` | ROC-AUC and accuracy (Y) vs \|corr(ŷ, gender)\| (X) for the λ sweep and the Keras Tuner trials, with the Pareto front |
| `figures/loss_curves.png` | Train/validation loss of every final training (Base, FAIR, error model) |
| `figures/uncertainty_by_class.png` | Uncertainty distribution for predicted good vs bad payers |
| `figures/uncertainty_by_missing_ext_sources.png` | Uncertainty vs number of missing EXT_SOURCE scores |
| `figures/error_model_calibration.png` | Predicted vs realised absolute error on test |
| `tables/base_vs_fair_test.{md,csv}` | Base vs best FAIR on test (best values highlighted) |
| `tables/candidates_val.csv` | Every sweep/tuner candidate with its validation metrics |
| `tables/test_predictions_uncertainty.csv` | Per-applicant class, probability, expected error, MC variance |
| `models/*.keras` | Trained Base, FAIR and error models |
| `run_summary.json` | Configuration, selected hyper-parameters and metrics |

## Architecture

See [docs/architecture.md](docs/architecture.md) for diagrams.
