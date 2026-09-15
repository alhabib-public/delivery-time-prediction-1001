import json

import matplotlib
matplotlib.use("Agg")

import pytest

from delivery import train
from delivery.exp import v001_hgb_vs_mlp_kfold5 as v001
from tests.conftest import requires_data


def test_exp_name_and_factories():
    assert v001.EXP_NAME == "v001_hgb_vs_mlp_kfold5"
    assert set(v001.make_candidates(0)) == {"hgb", "mlp"}
    assert v001.make_baseline_builder(0).methods == ("median", "ridge")
    assert v001._split_models("median,hgb") == (("median",), ("hgb",))
    assert v001._split_models(["ridge"]) == (("ridge",), ())
    with pytest.raises(ValueError):
        v001._split_models("nope")


@requires_data
def test_run_smoke(tmp_path):
    res = v001.run(n_rows=400, cv=2, models="median,ridge", output_dir=tmp_path, experiments=False,
                   diagnostics=False, figures=False)
    assert res.exp_name == "v001_hgb_vs_mlp_kfold5" and res.best_name == "ridge"
    assert (tmp_path / "COMPLETED_PREDICTIONS.csv").exists() and (tmp_path / "run_meta.json").exists()
    assert json.loads((tmp_path / "run_meta.json").read_text())["n_train"] == 400


@requires_data
def test_cli_wraps_experiment(tmp_path, capsys):
    rc = train.main(["--exp", "v001_hgb_vs_mlp_kfold5", "--models", "median", "--cv", "2", "--n-rows", "300",
                     "--output-dir", str(tmp_path), "--evaluate-only", "--no-experiments", "--no-figures"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== leaderboard" in out and "best model: median" in out and not (tmp_path / "COMPLETED_PREDICTIONS.csv").exists()
    assert train.main(["--exp", "does_not_exist"]) == 2
