
# Sparse Logistic Regression — Interpretation Notes

Person 1 · branch `person1/data-logistic` · model implemented in `src/02_logistic_model.py`

## Model specification

- **Target:** `Two_yr_Recidivism` (prevalence 0.4551 across all 6,172 rows)
- **Features:** `Number_of_Priors`, `Age_Above_FourtyFive`, `Age_Below_TwentyFive`, `Female`, `Misdemeanor`
- **Model:** L1-penalised logistic regression, `C = 0.03`, liblinear solver
- **Preprocessing:** `Number_of_Priors` standardised; the four binary indicators passed through unchanged
- **Splits:** frozen stratified 80/20, seeds 7 / 21 / 42 (4,937 train / 1,235 test)

`C` was selected by 5-fold stratified cross-validation **on training rows only**, over the grid
{0.01, 0.03, 0.1, 0.3, 1, 3}. `C = 0.03` gave the highest CV AUC (0.7200) and was also the
strongest regularisation retaining all five features. The held-out test set was scored once,
after `C` was frozen, and was never used for model selection.

## How to read the coefficients

**`Number_of_Priors` is standardised.** Its odds ratio is per **one standard deviation**, which
is **4.7312 priors** — not per single prior.

- Per standard deviation (~4.73 priors): odds ratio **2.05**
- Per single additional prior: exp(0.7179 / 4.7312) ≈ **1.16**

Reading the tabulated 2.05 as "each prior doubles the odds" overstates the effect roughly
fourfold. The four binary features were not scaled, so their odds ratios read directly.

**Reference categories.** Each binary feature is measured against the group coded 0:

| Feature | Coded 1 | Reference group (coded 0) |
|---|---|---|
| `Age_Above_FourtyFive` | over 45 | aged 25–45 |
| `Age_Below_TwentyFive` | under 25 | aged 25–45 |
| `Female` | female | male |
| `Misdemeanor` | misdemeanor charge | felony charge |

Both age indicators are 0 for people aged 25–45, so that band is the implicit baseline for
both age coefficients.

## Coefficient table (seed 42, C = 0.03)

| Feature | Coefficient | Odds ratio | Direction |
|---|---|---|---|
| `Number_of_Priors` | +0.7179 | 2.0501 (per SD) | more priors ↑ predicted risk |
| `Age_Below_TwentyFive` | +0.5265 | 1.6930 | under 25 ↑ vs 25–45 |
| `Age_Above_FourtyFive` | −0.5164 | 0.5967 | over 45 ↓ vs 25–45 |
| `Misdemeanor` | −0.2614 | 0.7700 | misdemeanor ↓ vs felony |
| `Female` | −0.2110 | 0.8098 | female ↓ vs male |

Intercept: −0.0123. All five coefficients are non-zero, so L1 selected no sparser model here.

## Test performance (seed 42)

| Metric | Value |
|---|---|
| ROC-AUC | 0.7302 |
| Accuracy | 0.6834 |
| Precision | 0.6871 |
| Recall | 0.5587 |
| F1 | 0.6163 |
| Brier score | 0.2119 |
| Confusion (tn / fp / fn / tp) | 530 / 143 / 248 / 314 |

Classes assigned at the default 0.5 probability threshold. Recall is noticeably lower than
precision: at this threshold the model misses more reoffenders than it falsely flags.

## Robustness across seeds

| Seed | ROC-AUC | Accuracy | F1 | Brier |
|---|---|---|---|---|
| 7 | 0.7170 | 0.6607 | 0.6006 | 0.2151 |
| 21 | 0.7202 | 0.6745 | 0.6082 | 0.2152 |
| 42 | 0.7302 | 0.6834 | 0.6163 | 0.2119 |

**AUC spread across seeds: 0.0131.** Performance is therefore sensitive to which rows land in
the test set, and single-split differences smaller than roughly 0.013 should not be treated as
meaningful.

Coefficient stability, same fixed `C = 0.03` refitted per seed:

| Feature | Mean | SD | Non-zero in | Sign stable |
|---|---|---|---|---|
| `Number_of_Priors` | +0.7354 | 0.0214 | 3/3 | yes |
| `Age_Below_TwentyFive` | +0.5539 | 0.0240 | 3/3 | yes |
| `Age_Above_FourtyFive` | −0.5407 | 0.0294 | 3/3 | yes |
| `Female` | −0.2199 | 0.0229 | 3/3 | yes |
| `Misdemeanor` | −0.2055 | 0.0553 | 3/3 | yes |

Every feature survives in every seed and no coefficient changes sign, so the qualitative
interpretation above is stable across splits. `Misdemeanor` is the least stable in relative
terms (SD is about 27% of its mean) and its magnitude should be treated as approximate.

## What these coefficients are not

These are **predictive associations within this dataset**, not causal effects. No confounding
was adjusted for and no causal identification strategy was used. "Being under 25 raises the
odds" means the model assigns higher predicted risk to people under 25 in these data — not
that age causes reoffending.

Nor is this an audit or reproduction of the proprietary COMPAS algorithm. The task is
prediction of observed two-year recidivism in a public dataset.

## Exclusions

- **`score_factor`** — dropped from `clean_compas.csv` entirely. It is derived from the COMPAS
  risk score, so it is model output rather than an ordinary pre-assessment feature; including
  it would leak the thing being predicted.
- **Race indicators** — retained in the clean dataset but excluded from the primary model.
  They belong to the secondary sensitivity analysis (P1-10), to be run only after the primary
  results are frozen.

## Limitations

1. **L1 is not performing feature selection here.** Every setting from `C = 0.03` upward keeps
   all five features, and the only setting that prunes anything (`C = 0.01`, which zeroes
   `Female`) costs about 1.5 AUC points. With five pre-selected features there is nothing for
   sparsity to remove; the "sparse" framing is not doing work at this feature-set size.
2. **Absolute performance is modest.** AUC ≈ 0.73 means that given one reoffender and one
   non-reoffender, the model ranks them correctly about 73% of the time.
3. **Seed sensitivity exceeds most differences of interest.** The 0.0131 AUC spread is larger
   than the gap between this model and the XGBoost comparison, so per-seed comparisons on a
   single split are not informative.
4. **Threshold is unexamined.** All class-based metrics use 0.5; no threshold tuning was done,
   and precision/recall trade-offs would shift considerably at other cutoffs.

## Preliminary comparison note

Against Person 2's XGBoost results on the same frozen splits (formal comparison is Day 7,
`05_compare_models.py`):

| Seed | Logistic AUC | XGBoost AUC | Gap |
|---|---|---|---|
| 7 | 0.7170 | 0.7233 | +0.0063 |
| 21 | 0.7202 | 0.7228 | +0.0026 |
| 42 | 0.7302 | 0.7321 | +0.0019 |

XGBoost is ahead on all three seeds, but by 0.0019–0.0063 — smaller than either model's own
variation across seeds (0.0131 here, ~0.009 for XGBoost). Both models also rank the same
features in the same order: mean |SHAP| puts `Number_of_Priors` far ahead, then
`Age_Below_TwentyFive`, `Age_Above_FourtyFive`, `Misdemeanor`, `Female` — matching the
ordering of the coefficient magnitudes above. Pending joint verification on Day 7.
