## ML Take-Home Assignment: Food Delivery Time Prediction

You are given historical data about food deliveries. Your task is to build a model that predicts **how long a delivery will take**.

### Data

Under `dataset/`, you can find:

- `train.csv` — training data **with** target column `Time_taken(min)`
- `test.csv` — test data **without** the target
- `Sample_Submission.csv` — example file showing the required submission format and column names (includes `ID` and `Time_taken(min)`)

The dataset includes features such as delivery partner attributes, restaurant & destination coordinates, order date/time, weather conditions, traffic density, vehicle/order type, and similar fields.

Like most real-world datasets, expect **missing values and noisy entries**.

### Task

1. Train one or more models using `train.csv`.
2. Use a validation approach of your choice (e.g., holdout split or cross-validation).
3. Generate predictions for every row in `test.csv`.

### Evaluation (what you should report)

- Use **MAE (Mean Absolute Error)** on your validation set as the primary metric. You are encouraged to use additional metrics and graphs that would make communicating
- If you apply target transformations (e.g., log), additional metrics, or specific split strategies, explain why.

### Deliverables

Please submit a single zip/folder containing:

1. **Short report** (PDF or Markdown, ~1–2 pages) including:
    - What you built and key assumptions
    - Validation setup + metrics
    - Summary of experiments (what you tried, what helped)
    - Error analysis (at least **2 slices**; e.g., traffic density buckets, distance buckets, city, weather, time-of-day)
    - Limitations and what you’d do next with more time
2. **Code** (notebook and/or scripts) runnable end-to-end:
    - load → preprocess → train → validate → predict on test
    - include a `README.md` with run instructions
3. **Predictions file**
    - A CSV matching the exact format of `Sample_Submission.csv` (same column names and ordering), containing predictions for `test.csv`.
4. **Dependencies**
    - `requirements.txt` (or equivalent)
5. **Tooling disclosure**
    - If you used LLMs or external tools, briefly note what you used them for.

### Time expectations

This should fit within **~4–8 hours** of focused work. Please submit within **48 hours**.