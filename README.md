## ML Assignment: Food Delivery Time Prediction

To read the full assignment submission which includes the report and pipeline results, please download and open the following HTML in your browser of choice: [COMPLETED_ASSIGNMENT.html](COMPLETED_ASSIGNMENT.html) 

### 1. Methodology and some comments

1. **This submission closely follows the sklearn guideline**, by preparing the processing, fitting, transforming, and evaluation steps as sklearn-compliant pipes.

2. Structuring everything into an sklearn pipeline **streamlines plug-and-playing things**, which enables quick iteration over experiments and ideas.

3. **The notebook generates all results via one pipeline** that runs all the necessary steps in sequence. This produces clean and scalable reports.

4. **Everything in this submission is runnable via Docker**. This is to sidestep all virtual environment and reproducibility issues.

5. **My dev-flow** when preparing this submission was: 

* Prepare **high and mid-level scaffolding as pseudo-code**, and use **Claude Fable 5 to produce the low-level code**. 
* Then I did several **self-review rounds to make sure everything is logically sound**, and the **results are clean and easy to follow**.

4. I really enjoyed this assignment so **many thanks to who prepared it** !


## Key assumptions and high-level implementaion comments

* Missing values arrive as the literal string `"NaN"`, every string has trailing whitespace.
  
* Test rows share the train date range and every test courier appears in train → the test set is an i.i.d. sample, so **shuffled K-fold CV** is the right validation.

* The target stays on its original scale. A log1p transform was tried as an experiment and did not help (the target is only mildly skewed).

* Impossible values are treated as missing rather than kept or dropped: ratings above 5, and the `0` / `0.01` coordinate placeholders (about 8 % of rows). Negative coordinates are sign-flipped, since all cities are in India.

* Missingness is informative, not random (rows with a missing field average 23.8 minutes vs 26.6 without) → missing values are kept as flags and handled natively by the model instead of dropping rows.

* Model selection uses out-of-fold MAE only; the test set is never looked at. The winner is refit once on all training rows before predicting `test.csv`.

* Everything is seeded (seed 0) and runs on CPU inside Docker, so every number in this report is reproducible with `make train`.


## 1. What was built

**Structure.** 
* One versioned experiment module (`delivery/exp/v001_hgb_vs_mlp_kfold5.py`)

* Each experiment makes a single call into a generic pipeline (`delivery/pipeline/regression.py::execute`) that
runs the experiment phases: validation → preparation → pre-analysis → baselines → cross-validation → holdout
diagnostics → experiments → refit + predict → dump → post-analysis.


**Model pipeline.** Everything the model needs lives inside one sklearn pipeline, so cross-validation cannot leak and `test.csv` goes through exactly the same steps as `train.csv`:

`DeliveryCleaner` → `DeliveryFeatureEngineer` → `ColumnTransformer` → model

1. **Cleaning.** The raw file is noisy: missing values are the text `"NaN"`, every string has trailing whitespace, ages and ratings are strings, and the target is written as `"(min) 24"`. The cleaner fixes all of this in one place. It also treats impossible values as missing: ratings above 5, and the `0` / `0.01` coordinate placeholders (about 8 % of rows). Negative coordinates are sign-flipped, since all the cities are in India.

2. **Features.** Engineered inside the pipeline from the cleaned columns:

| feature | built from | why |
|---|---|---|
| distance (km) | restaurant and customer coordinates (haversine) | the main physical driver of delivery time |
| order time, pickup time | the two timestamps, as minutes of the day | time-of-day effects |
| preparation time | pickup − order (with midnight wrap) | how long the restaurant took |
| weekday, weekend, day, month | order date | calendar effects |
| missing-value flags | one per field that can be missing | missingness itself is informative |
| courier city | prefix of the courier ID (22 cities) | city-level differences |
| age, rating, vehicle condition, extra deliveries, restaurant lat/lon | raw columns, as-is after cleaning | courier and location proxies |

   I deliberately did not use courier history or target encoding: it leaks inside cross-validation, and the test set is a random sample of the same period rather than the future.

3. **Encoding.** One encoder per model family, both sklearn `ColumnTransformer`s:

| model family | numeric features | categorical features | missing values |
|---|---|---|---|
| tree (gradient boosting, median) | as-is | integer codes, handled natively by the model | kept, handled natively |
| dense (Ridge, MLP) | median imputation + standardisation | one-hot (unknown level → all zeros) | imputed |

4. **Models.** Four candidates, same pipeline, same folds:

| candidate | model | loss | role |
|---|---|---|---|
| median | `DummyRegressor` | — | floor |
| ridge | `Ridge` | squared error | linear baseline |
| hgb | `HistGradientBoostingRegressor` | absolute error (= MAE, the metric) | main candidate, selected |
| mlp | `TorchMLPRegressor` (PyTorch, two hidden layers, early stopping) | Huber | neural candidate |

   I wrote the MLP to the sklearn estimator contract and verified it with sklearn's own `check_estimator` suite, so it drops into `Pipeline`, `clone` and `cross_validate` like any other estimator.

**Evaluation pipeline.** Also sklearn-style, and bespoke to this task:

* scorers built with `make_scorer`: MAE as the primary metric, plus RMSE, median absolute error, R², and the share of deliveries predicted within ±5 minutes, which is the number I would show a business stakeholder;
* a `DeliverySlicer` that maps every row to the error slices used below (traffic, distance bucket, city, time of day, weather, multiple deliveries);
* a `DeliveryEvaluator` that runs `cross_validate`, stitches the out-of-fold predictions together and produces the leaderboard, the per-fold scores and the per-slice tables;
* display objects in the sklearn `from_estimator / from_predictions / plot(ax=)` style, so the same figures work on any fitted estimator.


## 2. Validation setup and results

**Why shuffled 5-fold cross-validation.** `test.csv` covers the same dates as `train.csv` (11 Feb to 6 Apr 2022) and every test courier appears in the training data. So the test set is a random sample of the same distribution, not a future period, and a shuffled K-fold mirrors that. I still ran a courier-grouped split as a robustness check (Section 3).

**Every number in this report is out-of-fold.** Each training row is predicted by a model that never saw it. The 80/20 holdout in the diagnostics section below tells the same story.

| model | MAE | RMSE | MedAE | R² | within ±5 min | bias | MAE fold std |
|---|---|---|---|---|---|---|---|
| **gradient boosting** (selected) | **3.153** | 4.047 | 2.634 | 0.814 | 81.1 % | −0.20 | 0.030 |
| MLP (PyTorch, Huber loss) | 3.410 | 4.350 | 2.846 | 0.785 | 76.9 % | −0.06 | 0.026 |
| Ridge | 4.857 | 6.082 | 4.156 | 0.579 | 58.5 % | 0.00 | 0.024 |
| median baseline | 7.564 | 9.384 | 7.000 | −0.001 | 41.8 % | −0.32 | 0.038 |

**Takeaways:** 

1. **Gradient boosting wins** clearly: an MAE of about 3.2 minutes, with 81 % of deliveries predicted within 5 minutes of the truth, against a median-baseline MAE of 7.6.
2. **The ranking is stable**. Per-fold MAE for boosting is 3.12 / 3.12 / 3.18 / 3.19 / 3.16, and the MLP is about 0.25 minutes worse on every single fold.
3. The courier-grouped split gives 3.155 against 3.153, so **the model is not memorising couriers**.
4. **The selected model is refit** on all 41,093 rows before predicting `test.csv`. The predictions average 25.9 minutes against a training mean of 26.3, which is what I would expect from a random split.


## 3. Experiments: what I tried and what helped

I ran a small experiment ladder of 12 variants on the same folds (the full table is in the *[Analysis] Experiments* section below and in `outputs/experiments.json`). What I learned:

1. **The model family matters far more than anything else.** Boosting 3.15, MLP 3.41, linear 4.86, median 7.56. Delivery time is driven by interactions between traffic, distance and time of day, and trees pick those sufficiently.
2. **Boosting barely cares about tuning.** Out-of-the-box settings (squared loss, 100 iterations) land within 0.003 MAE of the tuned model. I kept the tuned absolute-error version because it is marginally best on the metric, but the honest reading is that the 3.15 floor comes from the features, not from the hyper-parameters.
4. **A log-transformed target does not help** (3.157 against 3.153). The target is only mildly skewed and the model already optimises an absolute-error loss on the raw scale, so the submission uses the raw target.
5. **Distance is the most valuable engineered feature.** Removing it costs 0.23 MAE. Removing the order and pickup time features changes nothing: the gap between them is a near-constant 5 to 15 minutes, and time of day is already captured by traffic density.

## 4. Error analysis

The figures and full tables are in the *[Analysis] Error analysis* section below.

* **Small note:** Bias is mean(predicted − actual), so a negative bias means the model under-predicts and vice versa.

1. **Traffic density is the strongest driver of both delivery time and error.** Low traffic is easy (MAE 2.7, 88 % within 5 minutes). Jams are the slowest bucket (31 minutes on average) and the most under-predicted (MAE 3.5, bias −0.45). The 529 rows with unknown traffic are the hardest of all at MAE 5.9.
2. **Error grows mildly with distance.** From MAE 2.8 under 5 km to 3.3 beyond 10 km, with under-prediction on the long trips. The 8 % of rows with placeholder coordinates sit at 3.8, which makes sense given that distance is the second most useful feature.
3. **Time of day matters.** The evening peak (18 to 21 h, 40 % of all orders) is the largest and the hardest known bucket at MAE 3.4; mornings are the easiest at 2.5. Orders with a missing order time are the single worst slice in the whole analysis (MAE 5.1, only 58 % within 5 minutes).
4. **Missing values are where the model struggles most.** Rows with unknown traffic, weather or order time all land at MAE 5 to 6 instead of 3. These fields tend to be missing together, so these are partially logged orders rather than three separate problems.
5. **The slow tails are handled well.** Festival days (45 minutes on average) and orders with three extra deliveries (48 minutes) have an MAE at or below 3.0, so the model captures those shifts through the features. City makes almost no difference, and weather is flat except when it is missing.

The pattern is simple: the residual error sits on (a) rows where the informative fields are missing, and (b) the slow deliveries (jams, evenings, long distances), which are systematically under-predicted by roughly half a minute because within those cells the true distribution has a long right tail that the available features cannot resolve.


## 5. Limitations and what I would do next

1. **The remaining error is mostly irreducible with these features.** Within a given traffic / distance / time cell the spread of delivery times is wide and nothing in the data explains it. The next real signal would be history: previous durations of the same courier or restaurant, or a live queue length.
3. **Hyper-parameters were hand-tuned on a small ladder, not searched.** A randomised search over the boosting and MLP settings is the obvious next step; the evaluation pipeline already exposes sklearn scorers for it.
4. **The MLP trails boosting by 0.25 minutes.** Untried options: longer training schedules, and averaging a few seeds, which is cheap and usually worth a small gain.

## 6. Tooling disclosure

* I used Claude Code (Claude Fable 5.1) as a pair-programming assistant. 
* As described in the methodology section: I prepared the high- and mid-level design, and Fable produced most of the low-level code and the tests.
* Every number in this report was produced by running the code in this repository (`make test`, `make train`, `make experiments`, and the pipeline cell below); nothing was estimated by hand. 

**Libraries used**: scikit-learn 1.9, PyTorch 2.14 (CPU), pandas 3.0, numpy 2.5, matplotlib 3.11.
