"""Evaluation outputs required by the evaluation contract (CLAUDE.md, items 4 and 8).

Scores are computed over the classes present in y_true. Labels that appear only in predictions (a held-out
tree predicted as another species) are kept in the confusion matrix but get no recall/F1 of their own.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from woodraman.io import REPO_ROOT


def classification_metrics(y_true, y_pred, groups=None) -> dict:
    """Balanced accuracy, macro-F1, per-class recall, confusion matrix; group vote accuracy if `groups`."""
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    classes = sorted(np.unique(y_true))
    labels = sorted(set(classes) | set(np.unique(y_pred)))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(cm, ([idx[t] for t in y_true], [idx[p] for p in y_pred]), 1)
    recall, f1 = {}, []
    for c in classes:
        tp = cm[idx[c], idx[c]]
        r = tp / cm[idx[c]].sum()
        col = cm[:, idx[c]].sum()
        prec = tp / col if col else 0.0
        recall[c] = float(r)
        f1.append(2 * prec * r / (prec + r) if prec + r else 0.0)
    out = {
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(f1)),
        "per_class_recall": recall,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "n": len(y_true),
        "chance": 1 / len(classes) if len(classes) > 1 else None,
    }
    if groups is not None:
        df = pd.DataFrame({"g": np.asarray(groups), "y": y_true, "p": y_pred})
        # majority vote over the spectra of one (group, true label) unit, e.g. the 30 spectra of a block
        vote = df.groupby(["g", "y"]).p.agg(lambda s: s.mode().iloc[0]).reset_index()
        out["group_vote_accuracy"] = float((vote.y == vote.p).mean())
        out["n_groups"] = len(vote)
    return out


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return proc.stdout.strip()
    except OSError:
        return ""


def run_config(script: str, **params) -> dict:
    """Provenance of one run: script, commit, date, versions, parameters.

    `script_clean` is False when the script itself is untracked or modified; `git_dirty` is True when any tracked
    file is modified OR the script is not clean (an uncommitted script cannot be reproduced from `git_commit`).
    """
    import scipy
    import sklearn

    script_clean = not _git("status", "--porcelain", "--untracked-files=all", "--", script)
    return {
        "script": script,
        "git_commit": _git("rev-parse", "HEAD"),
        "script_clean": script_clean,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")) or not script_clean,
        "date": datetime.now().isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "params": params,
    }


def write_run(out_dir: Path, metrics: dict, config: dict) -> None:
    """Write metrics.json and config.json into `out_dir` (created if missing)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
