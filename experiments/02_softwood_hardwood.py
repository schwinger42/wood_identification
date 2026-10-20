"""02 — Softwood vs hardwood on UNSEEN species measured on UNSEEN days.

    uv run python experiments/02_softwood_hardwood.py

Why this question: conifers (softwood) have guaiacyl (G) lignin; broadleaf trees (hardwood) have G + syringyl
(S) lignin. That is a known chemical difference with known Raman markers, and the data hold 4 softwood
species (5 trees) and 5 hardwood species (6 trees) — 11 independent trees instead of 4.

Split (strictest available): for each measurement date d, TEST = every species measured on d (on that day);
TRAIN = all spectra from other dates AND from other species. The model never sees the test species nor the
test session, so neither a species fingerprint nor a session shortcut can be memorised.

Controls: the same split on the silent region (1850–2700, no wood bands) and the O-H region.
Significance: exact permutation test over all C(9,4) = 126 ways to label 4 of the 9 species "softwood".
Writes results/<YYYYMMDD>_02_softwood_hardwood/{per_species.csv, summary.json}.
"""

from __future__ import annotations

import json
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from woodraman.io import REPO_ROOT, load_dataset
from woodraman.models import baseline_classifiers
from woodraman.preprocess import SNV, Crop, SGDerivative

SOFTWOOD = {
    "Chamaecyparis formosensis",
    "Chamaecyparis obtusa var. formosana",
    "Chamaecyparis obtusa",
    "Fokienia hodginsii",
}
REGIONS = {
    "fingerprint_400_1800": [(400, 1800)],
    "silent_1850_2700": [(1850, 2700)],
    "OH_3050_3500": [(3050, 3500)],
}
CLASSIFIERS = ("PCA-LDA", "LogReg", "PLS-DA")


def folds(meta: pd.DataFrame):
    """Leave-one-date-out, also removing the test species from training (unseen species + unseen session)."""
    for d in sorted(meta.date.unique()):
        test_species = set(meta.loc[meta.date == d, "species"])
        test = np.flatnonzero((meta.date == d).to_numpy())
        train = np.flatnonzero(((meta.date != d) & ~meta.species.isin(test_species)).to_numpy())
        yield d, train, test


def run(Z, y, meta, make_clf) -> pd.DataFrame:
    pred = np.empty(len(y), dtype=object)
    for _, train, test in folds(meta):
        pred[test] = make_clf().fit(Z[train], y[train]).predict(Z[test])
    df = meta[["species", "date", "tree_id", "block_id"]].copy()
    df["y"], df["p"] = y, pred
    return df


def score(df: pd.DataFrame) -> dict:
    per_sp = df.groupby("species").apply(lambda g: (g.p == g.y).mean(), include_groups=False)
    per_tree = df.groupby("tree_id").apply(lambda g: (g.p == g.y).mean(), include_groups=False)
    return {
        "spectrum_bal_acc": float(balanced_accuracy_score(df.y, df.p.astype(str))),
        "species_correct_by_majority": int((per_sp > 0.5).sum()),
        "trees_correct_by_majority": int((per_tree > 0.5).sum()),
        "per_species": per_sp.round(3).to_dict(),
    }


def main() -> None:
    X, wn, meta = load_dataset()
    clfs = baseline_classifiers()
    species = sorted(meta.species.unique())
    preps = {"SNV": None, "SG1+SNV": SGDerivative(1, 11, 2)}
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_02_softwood_hardwood"
    out.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    for pname, prep in preps.items():
        F = prep.transform(X) if prep is not None else X
        for region, win in REGIONS.items():
            Z = SNV().transform(Crop(wn, win).fit(F).transform(F))
            for cname in CLASSIFIERS:
                y = np.where(meta.species.isin(SOFTWOOD), "softwood", "hardwood")
                obs = score(run(Z, y, meta, clfs[cname]))
                if cname == "PCA-LDA":  # exact permutation test over species-level label assignments
                    null = []
                    for sw in combinations(species, 4):
                        yp = np.where(meta.species.isin(set(sw)), "softwood", "hardwood")
                        null.append(score(run(Z, yp, meta, clfs[cname]))["spectrum_bal_acc"])
                    null = np.array(null)
                    obs["perm_p"] = float((null >= obs["spectrum_bal_acc"] - 1e-12).mean())
                    obs["null_median"] = float(np.median(null))
                else:
                    obs["perm_p"] = float("nan")
                key = f"{pname}|{region}|{cname}"
                summary[key] = obs
                rows += [
                    {"preproc": pname, "region": region, "clf": cname, "species": s, "acc": a}
                    for s, a in obs["per_species"].items()
                ]
                print(
                    f"{key:45s} bal_acc {obs['spectrum_bal_acc']:.3f}  species {obs['species_correct_by_majority']}/9"
                    f"  trees {obs['trees_correct_by_majority']}/11  perm p {obs['perm_p']:.3f}",
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(out / "per_species.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
