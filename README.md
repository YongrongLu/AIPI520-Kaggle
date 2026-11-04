# Classical Music Meets Classical ML — Fall 2026

Predicting which past patrons of an orchestra will buy a 2014-15 season subscription
([Kaggle competition](https://www.kaggle.com/competitions/classical-music-meets-classical-ml-fall-2026), metric: AUROC).

| | AUROC |
| --- | --- |
| Cross-validation (5 seeds × stratified 5-fold) | 0.9876 |
| Public leaderboard | 0.97885 |

Final model: rank average of LightGBM (weight 3) and logistic regression (weight 2) on 67 account-level features.

## Repository layout

```
src/
  features.py          # data pipeline: six tables -> one feature row per account
  cv.py                # LightGBM / XGBoost / CatBoost / logistic regression, repeated stratified CV
  make_submission.py   # final LightGBM + LR blend -> outputs/submission.csv
  make_results.py      # regenerates every number in the writeup into results/
  ablate.py            # feature-group ablation (drop one group at a time)
  exp.py               # later experiments that did not improve CV (household, neighbour labels)
results/               # small CSVs + one chart backing the writeup (committed)
data/                  # put the Kaggle CSVs here (not committed)
outputs/               # predictions and submission (not committed)
```

## Results

| File | What it shows |
| --- | --- |
| `cv_summary.csv` | CV AUROC (mean ± sd over 5 seeds) for each model and the blends |
| `blend_weights.csv` | Top 10 blend weightings from the grid search |
| `lgbm_tuning.csv` | The nine LightGBM settings compared |
| `ablation.csv` | Feature-group ablation and the geography follow-up |
| `feature_build_up.csv`, `feature_gain.png` | CV AUROC as feature groups are added one at a time |
| `experiments.csv` | Ideas that did not improve CV |
| `leakage_check.csv` | CV without features that may postdate the 2014-15 season |
| `feature_dictionary.csv` | Description of the 67 final features |

No competition data, model files or predictions are committed.

## How to reproduce

1. Install dependencies: `pip install -r requirements.txt`
2. Download the competition data from Kaggle and unzip the CSVs from `for_students/` into `data/`
   (or set `DATA_DIR` to their folder).
3. Run:

```bash
python src/cv.py lgb lr          # ~2 min; prints CV AUROC per model and saves predictions
python src/make_submission.py    # writes outputs/submission.csv
python src/cv.py                 # optional: all four models and blends
python src/make_results.py       # optional, ~15 min: regenerates results/
```

## Approach in brief

- **Pipeline.** Files are read as Latin-1 and aggregated by `account.id`. Accounts absent from a table get `NaN`,
  so trees can tell "never subscribed" from zero. Subscription and ticket features use seasons up to 2013-14 only.
- **Features (67).** Subscription recency, streaks and tier trend (35); ticket purchases (14); donations (8);
  distance to the nearest venue (1); exact first-donation date (2); account-creation order decoded from the
  Salesforce ID, plus label-free ID-batch statistics (7).
- **Selection.** Removing one feature group at a time showed that 11 of 12 geographic features hurt CV, so only
  venue distance is kept.
- **Models.** Shallow, heavily subsampled LightGBM (7 leaves, 30% of features per tree) plus logistic regression
  for diversity, blended on ranks because AUROC depends only on ordering.
- **Validation.** Stratified 5-fold CV repeated over 5 seeds. Changes were adopted only when they beat the
  seed-to-seed standard deviation (~0.0006). The public leaderboard (~70 positives) was not used for tuning.

The full writeup covers each decision with its evidence.

## Data

The competition data is licensed CC BY-NC-SA 4.0 and must be downloaded from Kaggle; it is not included here.
