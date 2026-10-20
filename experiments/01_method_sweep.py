"""01 — Is it just the ML method?  uv run python experiments/01_method_sweep.py

Question: does any reasonable preprocessing x classifier combination generalise to held-out TREES using
wood chemistry (fingerprint region), while failing the negative controls (silent region, session)?

A better method should RAISE fingerprint held-out-tree accuracy and LOWER the controls.
A method that raises both is just a better shortcut learner.

Tasks (all group splits; hyperparameters fixed, untuned):
  pair_hot  : within-day species pairs (0925 Chamaecyparis, 0926 Swietenia), hold out one tree of the
              2-tree species; binary recall on that tree. Session is held constant -> cleanest test.
  hot9      : 9-class, hold out one tree of a 2-tree species; recall on that tree.
  date      : Fokienia blocks measured on 0922 and 0923; leave-one-block-out, predict the DATE.
              Balanced accuracy; 0.5 = no session signal (good).
Writes results/<YYYYMMDD>_01_method_sweep/{per_tree.csv, summary.csv}.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from woodraman.io import REPO_ROOT, load_dataset
from woodraman.models import baseline_classifiers
from woodraman.preprocess import (
    SNV,
    ASPLSBaseline,
    Crop,
    NuisanceProjector,
    SGDerivative,
    session_difference_basis,
)
from woodraman.splits import assert_no_leak, held_out_tree_folds, leave_one_group_out

REGIONS = {"fingerprint": [(400, 1800)], "silent": [(1850, 2700)]}
PAIRS = {
    "0925_Chamaecyparis": ("20250925", ["Chamaecyparis formosensis", "Chamaecyparis obtusa var. formosana"]),
    "0926_Swietenia": ("20250926", ["Swietenia mahagoni", "Swietenia macrophylla"]),
}


def stateless_views(X, wn) -> dict[tuple[str, str], np.ndarray]:
    """Precompute per-spectrum (stateless) preprocessing for every (preproc, region)."""
    full = {
        "raw": X,
        "asPLS": ASPLSBaseline(wn, lam=1e5).transform(X),
        "SG1": SGDerivative(1, 11, 2).transform(X),
        "SG2": SGDerivative(2, 15, 2).transform(X),
    }
    views = {}
    for region, win in REGIONS.items():
        crop = Crop(wn, win).fit(X)
        for name, F in full.items():
            views[(name, region)] = SNV().transform(crop.transform(F))
    return views


PREPROCS = {  # name -> (stateless view, use EPO)
    "SNV": ("raw", False),
    "asPLS+SNV": ("asPLS", False),
    "SG1+SNV": ("SG1", False),
    "SG2+SNV": ("SG2", False),
    "SNV+EPO": ("raw", True),
    "SG1+SNV+EPO": ("SG1", True),
}


def fit_predict(make_clf, Z, y, train, test, basis):
    Ztr, Zte = Z[train], Z[test]
    if basis is not None and len(basis):
        proj = NuisanceProjector(basis)
        Ztr, Zte = proj.transform(Ztr), proj.transform(Zte)
    return make_clf().fit(Ztr, y[train]).predict(Zte)


def main() -> None:
    X, wn, meta = load_dataset()
    views = stateless_views(X, wn)
    clfs = baseline_classifiers()
    fk = (meta.species == "Fokienia hodginsii").to_numpy()
    rows = []
    for pname, (view, use_epo) in PREPROCS.items():
        for region in REGIONS:
            Z = views[(view, region)]
            # EPO basis from Fokienia repeated sessions. Fokienia is never a test tree in hot9/pair_hot,
            # so using all of it there is not leakage; for the date task it is re-estimated in-fold.
            epo_all = session_difference_basis(Z[fk], meta[fk]) if use_epo else None
            for cname, make_clf in clfs.items():
                base = {"preproc": pname, "region": region, "clf": cname}
                y = meta.species.to_numpy()
                for tree, train, test in held_out_tree_folds(meta):
                    assert_no_leak(meta, train, test, "tree_id")
                    p = fit_predict(make_clf, Z, y, train, test, epo_all)
                    rows.append(
                        {**base, "task": "hot9", "unit": tree, "score": float((p == y[test][0]).mean())}
                    )
                for d, species in PAIRS.values():
                    m = ((meta.date == d) & meta.species.isin(species)).to_numpy()
                    sub = meta[m].reset_index(drop=True)
                    Zs, ys = Z[m], sub.species.to_numpy()
                    for tree, train, test in held_out_tree_folds(sub):
                        assert_no_leak(sub, train, test, "tree_id")
                        p = fit_predict(make_clf, Zs, ys, train, test, epo_all)
                        rows.append(
                            {
                                **base,
                                "task": "pair_hot",
                                "unit": tree,
                                "score": float((p == ys[test][0]).mean()),
                            }
                        )
                sub = meta[fk].reset_index(drop=True)
                Zf, yd = Z[fk], sub.date.to_numpy()
                pred = np.empty(len(yd), dtype=object)
                for train, test in leave_one_group_out(sub, "block_id"):
                    basis = session_difference_basis(Zf[train], sub.iloc[train]) if use_epo else None
                    pred[test] = fit_predict(make_clf, Zf, yd, train, test, basis)
                rows.append(
                    {
                        **base,
                        "task": "date",
                        "unit": "Fokienia LOBO",
                        "score": float(balanced_accuracy_score(yd, pred.astype(str))),
                    }
                )
            print(f"done {pname} / {region}", flush=True)

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_01_method_sweep"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_tree.csv", index=False)
    summ = df.groupby(["preproc", "clf", "region", "task"]).score.mean().unstack(["region", "task"])
    summ.columns = [f"{r}_{t}" for r, t in summ.columns]
    summ = summ[
        [
            c
            for c in [
                "fingerprint_pair_hot",
                "silent_pair_hot",
                "fingerprint_hot9",
                "silent_hot9",
                "fingerprint_date",
                "silent_date",
            ]
            if c in summ.columns
        ]
    ].round(3)
    summ["margin_pair"] = (summ.fingerprint_pair_hot - summ.silent_pair_hot).round(3)
    summ = summ.sort_values("fingerprint_pair_hot", ascending=False)
    summ.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
