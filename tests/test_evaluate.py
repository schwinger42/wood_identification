import json

import numpy as np
import pytest

from woodraman.evaluate import classification_metrics, run_config, write_run


def test_perfect_prediction():
    y = np.array(["a", "a", "b", "c"])
    m = classification_metrics(y, y)
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion_matrix"]["matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert m["chance"] == pytest.approx(1 / 3)


def test_balanced_accuracy_ignores_class_imbalance():
    y = np.array(["a"] * 8 + ["b"] * 2)
    p = np.array(["a"] * 10)
    m = classification_metrics(y, p)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["per_class_recall"] == {"a": 1.0, "b": 0.0}
    f1_a = 2 * 0.8 * 1.0 / (0.8 + 1.0)
    assert m["macro_f1"] == pytest.approx(f1_a / 2)


def test_confusion_rows_are_true_cols_predicted():
    m = classification_metrics(np.array(["a", "a", "b"]), np.array(["b", "a", "b"]))
    assert m["confusion_matrix"]["labels"] == ["a", "b"]
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [0, 1]]


def test_predicted_only_labels_appear_in_confusion_but_not_in_scores():
    # held-out tree: every test spectrum has one true species, predictions spread over others
    y = np.array(["a"] * 4)
    p = np.array(["a", "b", "c", "c"])
    m = classification_metrics(y, p)
    assert m["per_class_recall"] == {"a": 0.25}
    assert m["balanced_accuracy"] == pytest.approx(0.25)
    assert m["confusion_matrix"]["labels"] == ["a", "b", "c"]
    assert m["confusion_matrix"]["matrix"][0] == [1, 1, 2]
    assert m["chance"] is None  # one true class: chance is undefined


def test_group_vote_accuracy_uses_majority_per_group():
    # spectrum-level accuracy is 4/6; both groups are right by majority, so the vote must give 1.0
    y = np.array(["a", "a", "a", "b", "b", "b"])
    p = np.array(["a", "a", "b", "b", "b", "a"])
    g = np.array(["g1", "g1", "g1", "g2", "g2", "g2"])
    m = classification_metrics(y, p, groups=g)
    assert m["group_vote_accuracy"] == 1.0
    assert m["n_groups"] == 2


def test_write_run_writes_metrics_and_config(tmp_path):
    cfg = run_config("experiments/99_demo.py", seed=0, regions={"fingerprint": [[400, 1800]]})
    write_run(tmp_path, {"task": {"balanced_accuracy": 0.5}}, cfg)
    assert json.loads((tmp_path / "metrics.json").read_text())["task"]["balanced_accuracy"] == 0.5
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["script"] == "experiments/99_demo.py"
    assert saved["params"]["seed"] == 0
    assert {"git_commit", "git_dirty", "date", "versions"} <= saved.keys()
