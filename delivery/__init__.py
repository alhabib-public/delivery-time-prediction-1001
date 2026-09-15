"""Food delivery time prediction (take-home assignment).

Layers (ml-pipe style)
----------------------
exp/           versioned experiment modules; each ``run()`` makes one ``execute()`` call
pipeline/      ``execute``: the generic phase runner (validate → CV → diagnostics →
               experiments → refit/predict → dump → post-analysis)
analysis/      display helpers, builders (metrics, baselines) and post-analysis reporters
data           load the CSVs, parse the target, write the submission file
preprocessing  sklearn transformers: cleaning, feature engineering, encoding
models         ``TorchMLPRegressor`` (sklearn-compliant PyTorch MLP) + candidate registry
evaluation     bespoke sklearn-style evaluation (scorers, slices, displays, styled tables)
train          CLI wrapper around an experiment module; ``experiments`` = offline ladder
"""

__version__ = "0.2.0"
