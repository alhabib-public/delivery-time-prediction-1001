# delivery-time-prediction

Take-home assignment (`ASSIGNMENT.md`): predict food delivery time, MAE on validation,
predictions for `test.csv`. Bootstrapped from the user's `pytorch-workbook` cookie-cutter;
its Docker-only workflow and notebook conventions still apply. Read this before changing
anything.

## Environment — Docker only

- No local venv. Every command runs in the image via `make <target>` (`make help` lists
  them). The repo is mounted at `/app`; `dataset/` is inside it, so paths are identical on
  the host and in the container. `delivery.data.data_dir()` also honours `DELIVERY_DATA_DIR`.
- `requirements.txt` installs the CPU-only torch wheel (`--extra-index-url`). Rebuild
  (`make build`) after touching it. There is no GPU in the container: `device` is always `cpu`.
- No `curl`, `make` or `docker` inside the image; run python directly there.
- `.devcontainer/devcontainer.json` shares the same `Dockerfile`; keep it in sync if the
  Dockerfile or workspace layout changes.

## Where things live

Three layers, mirroring the user's ml-pipe experiment convention:

| path | role |
|---|---|
| `delivery/exp/vNNN_<slug>.py` | **experiment module**: docstring = cumulative changelog, `make_*` factories, `run(display_postanalysis=False, ...)` that makes exactly one `execute(...)` call; `EXP_NAME = Path(__file__).stem` |
| `delivery/pipeline/regression.py` | **`execute(...)`**: keyword groups (Input / Reporters / Builders / Cleaner / Model / Validation / Experiments / Flags / Storage), numbered `# StepN:` phases, returns `ExecutionResult`; `Experiment` dataclass for extra variants |
| `delivery/analysis/display.py` | `display_title / display_dataframe / display_figure / display_info_box / display_markdown`: IPython in notebooks, plain text elsewhere |
| `delivery/analysis/builders.py` | `MetricsBuilder.build(y_true, y_pred)`, `BaselineBuilder.build()` |
| `delivery/analysis/reporters.py` | section reporters with config-only constructors: pre-analysis ones take `display(X, y, features)`, post-analysis ones `display(result)`; composed by `PreAnalysisReporter` / `PostAnalysisReporter` |
| `delivery/data.py` | raw-string CSV loading, target parsing, validated submission writer |
| `delivery/preprocessing.py` | `DeliveryCleaner`, `DeliveryFeatureEngineer`, `make_preprocessor("tree"/"dense")` |
| `delivery/models.py` | `TorchMLPRegressor` (sklearn contract, passes `check_estimator`), `make_candidate` registry |
| `delivery/evaluation.py` | bespoke sklearn-style evaluation: scorers, `DeliverySlicer`, `DeliveryEvaluator`, `*Display`, `style_*` tables, `save_report` |
| `delivery/train.py` | CLI wrapper: `--exp` picks the module, calls its `run()`, prints text tables |
| `delivery/experiments.py` | the offline experiment ladder summarised in `REPORT.ipynb` (`make experiments`) |
| `tests/` | pytest suite; `make test` |
| `REPORT.ipynb` | **the deliverable**: the written report (markdown cell) + ONE code cell (`exp.run(display_preanalysis=True, display_postanalysis=True)`) whose outputs are the results; shipped executed |
| `COMPLETED_ASSIGNMENT.html` | static export of `REPORT.ipynb` (`make report`), the file a reviewer reads first; regenerate after `make run-report` |
| `COMPLETED_PREDICTIONS.csv` | the submission (`ID,Time_taken (min)`), written by `make train` / `make predict` |
| `README.md` | **verbatim copy of the markdown cell of `REPORT.ipynb`**, nothing else; regenerate it whenever the report text changes (`make readme`) |
| `outputs/` | generated artefacts; `metrics.json`, `experiments.json`, `run_meta.json`, `figures/` are tracked |

## Rules

- **Hands off anything "ignored".** Never read, edit, move, delete or reference files or
  folders under `ignored/` or whose name starts with `ignored-` / `ignored_` (anywhere in
  the repo). They are the user's private scratch space and are git-ignored on purpose.
- **Package first.** Put logic in `delivery/` with tests; `REPORT.ipynb` only calls an
  experiment module. When features or models change: `make test`, `make train`, then
  `make run-report` (re-executes `REPORT.ipynb` in place so its outputs are fresh) and
  update the report text if a number moved.
- **Experiments are versioned, never edited.** A change to what runs = a new
  `delivery/exp/vNNN_<slug>.py` (copy the previous one, change one thing, prepend a
  "vNNN on top of vMMM: change + why; everything else unchanged" paragraph). `v001` stays
  as the reference arm quoted in `REPORT.ipynb`. New display sections = a new reporter in
  `analysis/reporters.py` wired into `PreAnalysisReporter` (data, before any fit) or
  `PostAnalysisReporter` (results); new phases go in `execute`.
- **sklearn conventions.** Transformers/estimators follow the sklearn 1.9 contract
  (`validate_data`, `__sklearn_tags__`, `check_is_fitted`, `clone`-safe `__init__`; never
  define `_more_tags`). Keep `TorchMLPRegressor` passing `check_estimator` (tested).
- **Never hand back a notebook without executing it** (`make run-report`); the report's
  markdown is the user's text — edit it only when asked, and never touch its numbers by hand.
- **Reviewer-facing repo.** Keep the root to what a reviewer needs;
  scratch files go under `ignored/` or an `ignored-` prefix.
- **Numbers are regenerated, never typed.** `REPORT.ipynb` quotes `outputs/metrics.json` and
  `outputs/experiments.json`; re-run and update the report after any change that moves them.
- pandas 3 in the image defaults to the string dtype: use
  `pd.api.types.is_numeric_dtype(s)`, not `s.dtype == object`.

## Timings (CPU, laptop)

`make test` ~15 s · `make train` ~1.5 min · `make run-report` ~2 min ·
`make experiments` ~7 min. Container clocks can jump if the Docker VM is paused; if
`fit_time_s` looks absurd, re-run.
