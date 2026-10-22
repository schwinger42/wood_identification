"""04 — What is the shared non-chemical pattern made of?  uv run python experiments/04_shared_pattern.py

One shape explains most of every spectrum (experiments/00). This script measures four properties of it that were
previously stated without a committed script:
(a) Day stability: does each day's median spectrum match the global median, and is the axis shifted?
(b) Peaks of the global median: which sit near expected wood bands, which in the silent region, is 1600 a dip?
(c) Scale vs shape: can the pattern's AMPLITUDE alone (one number per spectrum) predict species?
(d) Which spatial scale carries the silent-region shortcut: narrow structure or broad background?
Inputs to every classifier are spectra only; date is used as a split, a grouping or a control TARGET, never as
a feature. Writes results/<YYYYMMDD>_04_shared_pattern/{peaks.csv, day_stability.csv, metrics.json, config.json}.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from woodraman.evaluate import classification_metrics, run_config, write_run
from woodraman.io import REPO_ROOT, load_dataset
from woodraman.preprocess import SNV, Crop
from woodraman.splits import assert_no_leak, leave_one_group_out

WINDOW = (400, 3450)
SILENT = (1850, 2700)
# Same 13 band centres as experiments/02 (CLAUDE.md spectroscopy notes).
WOOD_BANDS = [380, 520, 1095, 1120, 1150, 1270, 1330, 1378, 1460, 1600, 1660, 2895, 2940]
MATCH_TOL = 20.0  # cm-1; chosen, not derived: C-H bands are broad, fingerprint bands ~10-20 cm-1 wide
SHIFTS = np.round(np.arange(-3.0, 3.0001, 0.05), 2)  # cm-1, candidate axis shifts per day
# cm-1 boxcar widths on a 1 cm-1 grid: "broad" = slower than the width, "narrow" = the rest. Several widths,
# because the answer depends on where the line is drawn (sensitivity, not a single choice).
BROAD_WIDTHS = (61, 121, 241)


class PatternAmplitude(BaseEstimator, TransformerMixin):
    """One feature per spectrum: least-squares amplitude of the TRAINING median shape (both mean-centred)."""

    def fit(self, X, y=None):
        g = np.median(np.asarray(X, dtype=np.float64), axis=0)
        self.shape_ = g - g.mean()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - X.mean(axis=1, keepdims=True)
        return (Xc @ self.shape_ / (self.shape_ @ self.shape_))[:, None]


def lobo(make_model, Z, y, meta) -> dict:
    pred = np.empty(len(y), dtype=object)
    for train, test in leave_one_group_out(meta, "block_id"):
        assert_no_leak(meta, train, test, "block_id")
        pred[test] = make_model().fit(Z[train], y[train]).predict(Z[test])
    return classification_metrics(y, pred, groups=meta.block_id.to_numpy())


def pca_lda():
    return make_pipeline(PCA(20, svd_solver="full"), LinearDiscriminantAnalysis())


def day_stability(X, wn, meta) -> pd.DataFrame:
    k = (wn >= WINDOW[0]) & (wn <= WINDOW[1])
    g = np.median(X[:, k], axis=0)
    rows = []
    for d in sorted(meta.date.unique()):
        m = np.median(X[(meta.date == d).to_numpy()], axis=0)
        corr = [np.corrcoef(g, np.interp(wn[k] + s, wn, m))[0, 1] for s in SHIFTS]
        rows.append(
            {
                "date": d,
                "corr_with_global_median": float(np.corrcoef(g, m[k])[0, 1]),
                "best_shift_cm1": float(SHIFTS[int(np.argmax(corr))]),
                "corr_at_best_shift": float(np.max(corr)),
            }
        )
    return pd.DataFrame(rows)


def peaks(X, wn) -> tuple[pd.DataFrame, list[float]]:
    k = (wn >= WINDOW[0]) & (wn <= WINDOW[1])
    g, w = np.median(X[:, k], axis=0), wn[k]
    pk, prop = find_peaks(g, prominence=0)
    bands = np.array(WOOD_BANDS, dtype=float)
    rows = []
    for i, p in enumerate(pk):
        nearest = bands[np.argmin(np.abs(bands - w[p]))]
        rows.append(
            {
                "position_cm1": round(float(w[p]), 1),
                "prominence": round(float(prop["prominences"][i]), 1),
                "in_silent_region": bool(SILENT[0] <= w[p] <= SILENT[1]),
                "nearest_wood_band": float(nearest),
                "offset_cm1": round(float(w[p] - nearest), 1),
                "near_wood_band": bool(abs(w[p] - nearest) <= MATCH_TOL),
            }
        )
    df = pd.DataFrame(rows).sort_values("prominence", ascending=False).reset_index(drop=True)
    dips, _ = find_peaks(-g, prominence=100)
    dips_1600 = [round(float(w[d]), 1) for d in dips if abs(w[d] - 1600) <= 20]
    return df, dips_1600


def scale_split(X, wn, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Silent region, SNV per spectrum, on a 1 cm-1 grid; broad = boxcar mean of `width`, narrow = residual."""
    grid = np.arange(SILENT[0], SILENT[1] + 1, 1.0)
    Z = SNV().transform(np.array([np.interp(grid, wn, x) for x in X]))
    broad = uniform_filter1d(Z, width, axis=1, mode="nearest")
    return broad, Z - broad


def main() -> None:
    X, wn, meta = load_dataset()
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_04_shared_pattern"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    y = meta.species.to_numpy()
    fk = (meta.species == "Fokienia hodginsii").to_numpy()
    fk_meta = meta[fk].reset_index(drop=True)
    metrics: dict = {}

    # (a) day stability ----------------------------------------------------------------------------
    stab = day_stability(X, wn, meta)
    stab.to_csv(out / "day_stability.csv", index=False)
    print(stab.round(3).to_string(index=False), "\n")
    metrics["day_stability"] = stab.to_dict(orient="records")

    # (b) peaks of the global median -----------------------------------------------------------------
    pk, dips_1600 = peaks(X, wn)
    pk.to_csv(out / "peaks.csv", index=False)
    print(pk.head(20).to_string(index=False))
    print(f"dips within 20 cm-1 of 1600: {dips_1600}\n")
    metrics["peaks_top20"] = pk.head(20).to_dict(orient="records")
    metrics["dips_near_1600"] = dips_1600

    # (c) amplitude of the shared pattern alone -> species, LOBO -------------------------------------
    def amp():
        return make_pipeline(Crop(wn, [WINDOW]), PatternAmplitude(), LinearDiscriminantAnalysis())

    metrics["amplitude_only_species_LOBO"] = lobo(amp, X, y, meta)

    # (d) narrow vs broad structure in the silent region --------------------------------------------
    for width in BROAD_WIDTHS:
        broad, narrow = scale_split(X, wn, width)
        for name, Z in (("broad", broad), ("narrow", narrow)):
            metrics[f"silent_{name}{width}_species_LOBO"] = lobo(pca_lda, Z, y, meta)
            metrics[f"silent_{name}{width}_fokienia_date_LOBO"] = lobo(
                pca_lda, Z[fk], fk_meta.date.to_numpy(), fk_meta
            )

    for key in ("amplitude_only_species_LOBO", *[k for k in metrics if k.startswith("silent_")]):
        m = metrics[key]
        print(f"{key:40s} balanced acc {m['balanced_accuracy']:.3f}  chance {m['chance']:.3f}")
    config = run_config(
        "experiments/04_shared_pattern.py",
        window=WINDOW,
        silent=SILENT,
        wood_bands=WOOD_BANDS,
        match_tol_cm1=MATCH_TOL,
        shifts_cm1=[float(SHIFTS[0]), float(SHIFTS[-1]), 0.05],
        broad_widths_cm1=BROAD_WIDTHS,
        pca_components=20,
    )
    write_run(out, metrics, config)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
