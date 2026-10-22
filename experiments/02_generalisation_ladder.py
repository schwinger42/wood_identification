"""02 — How far does each claim generalise?  uv run python experiments/02_generalisation_ladder.py

(a) 9-class species, PCA-LDA, SNV: spot-level random 5-fold -> leave-one-block-out -> held-out tree,
    fingerprint vs silent region. (Spot-level random CV is reported ONLY to show the inflation.)
(b) Softwood vs hardwood (chemically the easiest split: S-lignin is in hardwoods): leave-one-tree-out vs
    leave-one-DAY-out (test day unseen; trees seen on that day are also dropped from training), plus an exact
    permutation test of the day split: every one of the C(9,4) = 126 ways to call 4 species "softwood".
(c) Known wood-band features vs the same features at fake positions (silent region, off-band fingerprint),
    within-day pairs, leave-one-block-out.
Exploratory: several analyses were tried after seeing earlier results; treat as hypothesis-generating.
Writes results/<YYYYMMDD>_02_generalisation_ladder/{ladder.csv, soft_hard.csv, soft_hard_permutation.csv,
band_features.csv, metrics.json, config.json}.
"""

from __future__ import annotations

from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from woodraman.evaluate import classification_metrics, run_config, write_run
from woodraman.io import REPO_ROOT, load_dataset
from woodraman.models import baseline_classifiers
from woodraman.preprocess import SNV, Crop
from woodraman.splits import held_out_tree_folds, leave_one_group_out

REGIONS = {
    "fingerprint": [(400, 1800)],
    "silent": [(1850, 2700)],
    "C-H": [(2800, 3050)],
    "O-H": [(3050, 3500)],
}
SOFTWOODS = {
    "Chamaecyparis formosensis",
    "Chamaecyparis obtusa var. formosana",
    "Chamaecyparis obtusa",
    "Fokienia hodginsii",
}
WOOD_BANDS = [380, 520, 1095, 1120, 1150, 1270, 1330, 1378, 1460, 1600, 1660, 2895, 2940]
PAIRS = {
    "0925 C.formosensis vs C.obtusa var.formosana": (
        "20250925",
        ["Chamaecyparis formosensis", "Chamaecyparis obtusa var. formosana"],
    ),
    "0926 S.mahagoni vs S.macrophylla": ("20250926", ["Swietenia mahagoni", "Swietenia macrophylla"]),
}


def view(X, wn, region):
    return SNV().transform(Crop(wn, REGIONS[region]).fit(X).transform(X))


def pca_lda():
    return make_pipeline(PCA(20, svd_solver="full"), LinearDiscriminantAnalysis())


def ladder(X, wn, meta) -> tuple[pd.DataFrame, dict]:
    y = meta.species.to_numpy()
    blocks = meta.block_id.to_numpy()
    rows, metrics = [], {}
    for region in ("fingerprint", "silent"):
        Z = view(X, wn, region)
        p = cross_val_predict(pca_lda(), Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        rows.append(
            {"region": region, "split": "spot-level random 5-fold", "score": balanced_accuracy_score(y, p)}
        )
        metrics[f"{region}|spot-level random 5-fold"] = classification_metrics(y, p, groups=blocks)
        p = np.empty(len(y), dtype=object)
        for tr, te in leave_one_group_out(meta, "block_id"):
            p[te] = pca_lda().fit(Z[tr], y[tr]).predict(Z[te])
        rows.append(
            {
                "region": region,
                "split": "leave-one-block-out",
                "score": balanced_accuracy_score(y, p.astype(str)),
            }
        )
        metrics[f"{region}|leave-one-block-out"] = classification_metrics(y, p, groups=blocks)
        rec = []
        for tree, tr, te in held_out_tree_folds(meta):
            pt = pca_lda().fit(Z[tr], y[tr]).predict(Z[te])
            rec.append((pt == y[te][0]).mean())
            metrics[f"{region}|held-out tree|{tree}"] = classification_metrics(y[te], pt)
        rows.append(
            {"region": region, "split": "held-out tree (mean recall, 4 trees)", "score": float(np.mean(rec))}
        )
    return pd.DataFrame(rows), metrics


def soft_hard_folds(meta, kind):
    """Leave-one-tree-out, or leave-one-day-out with every tree measured on the test day dropped from training."""
    units = meta.tree_id.unique() if kind == "tree" else sorted(meta.date.unique())
    for u in units:
        if kind == "tree":
            te = np.flatnonzero(meta.tree_id == u)
            tr = np.flatnonzero(meta.tree_id != u)
        else:
            te = np.flatnonzero(meta.date == u)
            trees = set(meta.tree_id.iloc[te])
            tr = np.flatnonzero((meta.date != u) & ~meta.tree_id.isin(trees))
        yield tr, te


def soft_hard(X, wn, meta) -> tuple[pd.DataFrame, dict]:
    y = np.where(meta.species.isin(SOFTWOODS), "softwood", "hardwood")
    clfs = baseline_classifiers()
    rows, metrics = [], {}
    for region in REGIONS:
        Z = view(X, wn, region)
        for cname in ("PCA-LDA", "PLS-DA", "LogReg", "LinearSVM"):
            for kind in ("tree", "day"):
                p = np.empty(len(y), dtype=object)
                for tr, te in soft_hard_folds(meta, kind):
                    p[te] = clfs[cname]().fit(Z[tr], y[tr]).predict(Z[te])
                rows.append(
                    {
                        "region": region,
                        "clf": cname,
                        "split": f"leave-one-{kind}-out",
                        "bacc": balanced_accuracy_score(y, p.astype(str)),
                    }
                )
                metrics[f"{region}|{cname}|leave-one-{kind}-out"] = classification_metrics(
                    y, p, groups=meta.tree_id.to_numpy()
                )
    return pd.DataFrame(rows), metrics


def soft_hard_permutation(X, wn, meta) -> pd.DataFrame:
    """Leave-one-day-out softwood vs hardwood with PCA-LDA, under every relabelling of 4 of the 9 species as
    "softwood". p = share of the 126 labellings (the real one included) scoring >= the real labels."""
    species = sorted(meta.species.unique())
    folds = list(soft_hard_folds(meta, "day"))
    rows = []
    for region in REGIONS:
        Z = view(X, wn, region)

        def score(softwoods, Z=Z):
            y = np.where(meta.species.isin(softwoods), "softwood", "hardwood")
            p = np.empty(len(y), dtype=object)
            for tr, te in folds:
                p[te] = pca_lda().fit(Z[tr], y[tr]).predict(Z[te])
            return classification_metrics(y, p)["balanced_accuracy"]

        obs = score(SOFTWOODS)
        null = np.array([score(set(sw)) for sw in combinations(species, len(SOFTWOODS))])
        rows.append(
            {
                "region": region,
                "observed_bacc": obs,
                "perm_p": float((null >= obs - 1e-12).mean()),
                "null_median": float(np.median(null)),
                "n_labellings": len(null),
            }
        )
    return pd.DataFrame(rows)


def band_features(X, wn, meta) -> tuple[pd.DataFrame, dict]:
    grid = np.arange(380, 3450, 1.0)
    G = SNV().transform(np.array([np.interp(grid, wn, x) for x in X]))

    def feats(centers, half=12, gap=20, side=15):
        cols = []
        for c in centers:
            core = (grid >= c - half) & (grid <= c + half)
            sides = ((grid >= c - gap - side) & (grid < c - gap)) | (
                (grid > c + gap) & (grid <= c + gap + side)
            )
            cols.append(G[:, core].mean(1) - G[:, sides].mean(1))
        return np.c_[cols].T

    rng = np.random.default_rng(0)
    far = np.array([g for g in np.arange(420, 1780, 1.0) if min(abs(g - b) for b in WOOD_BANDS) >= 40])
    sets = {"wood bands": WOOD_BANDS, "fake: silent": np.linspace(1900, 2650, len(WOOD_BANDS))}
    for i in range(5):
        sets[f"fake: off-band fingerprint #{i}"] = np.sort(rng.choice(far, len(WOOD_BANDS), replace=False))
    rows, metrics = [], {}
    for name, cs in sets.items():
        F = feats(cs)
        for pname, (d, sp) in PAIRS.items():
            m = ((meta.date == d) & meta.species.isin(sp)).to_numpy()
            sub = meta[m].reset_index(drop=True)
            ys = sub.species.to_numpy()
            p = np.empty(len(ys), dtype=object)
            for tr, te in leave_one_group_out(sub, "block_id"):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
                p[te] = clf.fit(F[m][tr], ys[tr]).predict(F[m][te])
            rows.append(
                {"features": name, "pair": pname, "LOBO_bacc": balanced_accuracy_score(ys, p.astype(str))}
            )
            metrics[f"{name}|{pname}"] = classification_metrics(ys, p, groups=sub.block_id.to_numpy())
    return pd.DataFrame(rows), metrics


def main() -> None:
    X, wn, meta = load_dataset()
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_02_generalisation_ladder"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    a, m_a = ladder(X, wn, meta)
    a.to_csv(out / "ladder.csv", index=False)
    print(a.round(3).to_string(index=False), "\n")
    b, m_b = soft_hard(X, wn, meta)
    b.to_csv(out / "soft_hard.csv", index=False)
    print(b.groupby(["split", "region"]).bacc.agg(["mean", "min", "max"]).round(3).to_string(), "\n")
    perm = soft_hard_permutation(X, wn, meta)
    perm.to_csv(out / "soft_hard_permutation.csv", index=False)
    print(perm.round(3).to_string(index=False), "\n")
    c, m_c = band_features(X, wn, meta)
    c.to_csv(out / "band_features.csv", index=False)
    print(c.round(3).to_string(index=False))
    metrics = {
        "ladder": m_a,
        "soft_hard": m_b,
        "soft_hard_permutation": perm.to_dict(orient="records"),
        "band_features": m_c,
    }
    config = run_config(
        "experiments/02_generalisation_ladder.py",
        regions=REGIONS,
        softwoods=sorted(SOFTWOODS),
        wood_bands=WOOD_BANDS,
        pairs=PAIRS,
        pca_components=20,
        seeds={"random_5fold": 0, "fake_band_positions": 0},
    )
    write_run(out, metrics, config)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
