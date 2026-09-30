# Milestone versions

This folder contains four notebooks from the 13 submitted versions of the project, chosen because each marks a real change in approach. The final pipeline (V23, best test score) is in [`../notebook/heavy_equipment_price_prediction.ipynb`](../notebook/heavy_equipment_price_prediction.ipynb).

| Notebook | Version | What changed | CV RMSLE | Test RMSLE |
|---|---|---|---:|---:|
| [`v08_catboost_native_categoricals.ipynb`](v08_catboost_native_categoricals.ipynb) | V8 | Baseline: CatBoost with native handling of 42 categorical columns | 0.20843 | 0.20456 |
| [`v15_three_model_blend.ipynb`](v15_three_model_blend.ipynb) | V15 | First ensemble: XGBoost + LightGBM + CatBoost, weights tuned on OOF predictions | 0.19343 * | 0.19596 |
| [`v16_leakfree_target_encoding_stacking.ipynb`](v16_leakfree_target_encoding_stacking.ipynb) | V16 | In-fold smoothed target encoding, parsed features, Ridge stacking | 0.19990 | 0.19479 |
| [`v25_leakfree_pipeline_tuning.ipynb`](v25_leakfree_pipeline_tuning.ipynb) | V25 | Fully leak-free scikit-learn `Pipeline`, including during tuning | 0.19843 | 0.19320 † |

\* Optimistic: target encoding was fitted on the full training set before cross-validation.
† Submitted after the deadline.

These are the original Kaggle notebooks. Their code and outputs are unchanged apart from an added header cell (and one failed plotting cell removed from V8). Their data paths point to `/kaggle/input/...`.
