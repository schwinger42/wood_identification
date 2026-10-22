"""05 — Where do the linear models put their weight?  uv run python experiments/05_model_weights.py

For a linear classifier the decision score is linear in the spectrum, so each Raman-shift point i contributes
w_i * (x_i - mean_i) to it. The mean |contribution| per point is exactly what SHAP reports for a linear model
with independent features, so no SHAP/LIME is needed (and both are unstable here: neighbouring points move together).

Models (PCA(20)-LDA on SNV spectra, 400–3450 cm-1, same as the audit):
  species   : 9 classes, refit in each leave-one-block-out training fold (34 profiles -> median and 10–90% band)
  soft_hard : softwood vs hardwood, same folds
  date      : Fokienia 0922 vs 0923 (the only same-wood two-day data); a negative-control TARGET, never a feature
Questions: what share of each model's weight sits in the silent region, near known wood bands, or on the big peaks
that match no wood band, or in the vendor-baseline anchor windows? Does the species profile resemble the date profile MORE than it resembles where the
spectra simply vary most (per-point SD)? If not, the resemblance says nothing.
Writes results/<YYYYMMDD>_05_model_weights/{profiles.csv, model_weights.png, metrics.json, config.json}.
"""

from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from woodraman.evaluate import run_config, write_run
from woodraman.io import REPO_ROOT, load_dataset
from woodraman.preprocess import SNV, Crop
from woodraman.splits import leave_one_group_out

WINDOW = (400, 3450)
REGIONS = {"fingerprint": (400, 1800), "silent": (1850, 2700), "C-H": (2800, 3050), "O-H": (3050, 3450)}
WOOD_BANDS = [
    520,
    1095,
    1120,
    1150,
    1270,
    1330,
    1378,
    1460,
    1600,
    1660,
    2895,
    2940,
]  # 380 lies outside WINDOW
NONBAND_PEAKS = [
    648,
    832,
    1016,
]  # largest peaks of the global median; no wood band within 20 cm-1 (experiments/04)
NEAR = 20.0  # cm-1 half-width around a band or peak; chosen, as in experiments/04
# Windows where 78% / 81% of spectra are pinned <= 0: likely the vendor baseline's fixed anchors (experiments/00)
VENDOR_ANCHORS = [(955, 980), (2810, 2832)]
SOFTWOODS = {
    "Chamaecyparis formosensis",
    "Chamaecyparis obtusa var. formosana",
    "Chamaecyparis obtusa",
    "Fokienia hodginsii",
}


def contribution_profile(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean |w_ci * (z_i - mean_i)| over spectra and classes for PCA(20)-LDA fitted on (Z, y)."""
    pca = PCA(20, svd_solver="full").fit(Z)
    lda = LinearDiscriminantAnalysis().fit(pca.transform(Z), y)
    W = lda.coef_ @ pca.components_  # decision scores in the original Raman-shift coordinates
    D = np.abs(Z - Z.mean(axis=0))
    return np.abs(W).mean(axis=0) * D.mean(axis=0)


def shares(profile: np.ndarray, wn: np.ndarray, width: np.ndarray) -> dict:
    """Share of the total contribution (integrated over cm-1) per region, and the share of the axis it covers."""
    total, span = (profile * width).sum(), width.sum()
    masks = {name: (wn >= lo) & (wn <= hi) for name, (lo, hi) in REGIONS.items()}
    masks["near wood bands"] = np.min(np.abs(wn[:, None] - np.array(WOOD_BANDS)), axis=1) <= NEAR
    masks["on non-band peaks"] = np.min(np.abs(wn[:, None] - np.array(NONBAND_PEAKS)), axis=1) <= NEAR
    masks["vendor anchors"] = np.any([(wn >= lo) & (wn <= hi) for lo, hi in VENDOR_ANCHORS], axis=0)
    out = {}
    for name, m in masks.items():
        s, a = float((profile[m] * width[m]).sum() / total), float(width[m].sum() / span)
        out[name] = {"weight_share": s, "axis_share": a, "ratio": s / a}
    return out


def main() -> None:
    X, wn_all, meta = load_dataset()
    crop = Crop(wn_all, [WINDOW]).fit(X)
    wn = wn_all[crop.mask_]
    Z = SNV().transform(crop.transform(X))
    width = np.gradient(wn)  # the axis is non-uniform (1.01–2.59 cm-1): weight points by the cm-1 they cover
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_05_model_weights"
    out.mkdir(parents=True, exist_ok=True)

    targets = {
        "species": meta.species.to_numpy(),
        "soft_hard": np.where(meta.species.isin(SOFTWOODS), "softwood", "hardwood"),
    }
    folds = list(leave_one_group_out(meta, "block_id"))
    prof, band, fold_ratio = {}, {}, {}
    for name, y in targets.items():
        P = np.array([contribution_profile(Z[tr], y[tr]) for tr, _ in folds])
        prof[name] = np.median(P, axis=0)
        band[name] = np.percentile(P, [10, 90], axis=0)
        per_fold = [
            shares(p, wn, width) for p in P
        ]  # spread of each region's ratio over the 34 training sets
        fold_ratio[name] = {
            r: [float(q) for q in np.percentile([f[r]["ratio"] for f in per_fold], [10, 90])]
            for r in per_fold[0]
        }
    fk = (meta.species == "Fokienia hodginsii").to_numpy()
    prof["date"] = contribution_profile(Z[fk], meta.date.to_numpy()[fk])
    prof["spectral SD"] = Z.std(axis=0)

    metrics: dict = {name: shares(p, wn, width) for name, p in prof.items()}
    metrics["ratio_10_90_over_folds"] = fold_ratio
    top = {}
    for name in ("species", "soft_hard", "date"):
        pk, _ = find_peaks(prof[name], distance=15)
        pk = pk[np.argsort(prof[name][pk])[::-1][:5]]
        top[name] = [
            {"cm1": round(float(wn[i]), 1), "rel_weight": round(float(prof[name][i] / prof[name].max()), 2)}
            for i in pk
        ]
    metrics["top_weight_positions"] = top

    def rho(a: str, b: str) -> float:
        return float(spearmanr(prof[a], prof[b]).statistic)

    metrics["spearman"] = {
        "species~date": rho("species", "date"),
        "species~spectral SD": rho("species", "spectral SD"),
        "soft_hard~date": rho("soft_hard", "date"),
        "soft_hard~spectral SD": rho("soft_hard", "spectral SD"),
        "date~spectral SD": rho("date", "spectral SD"),
    }
    pd.DataFrame({"raman_shift_cm1": wn, **prof}).to_csv(out / "profiles.csv", index=False)

    rows = [{"profile": p, "region": r, **v} for p in prof for r, v in metrics[p].items()]
    pd.set_option("display.width", 200)
    print(
        pd.DataFrame(rows).pivot(index="region", columns="profile", values="ratio").round(2).to_string(),
        "\n(ratio = share of weight / share of the axis; 1 = no preference)\n",
    )
    for name, rr in fold_ratio.items():
        print(name, "ratio 10-90% over folds:", {r: [round(q, 2) for q in v] for r, v in rr.items()})
    print({k: round(v, 3) for k, v in metrics["spearman"].items()})
    for name, t in top.items():
        print(name, "top weights at", [d["cm1"] for d in t])

    # Figure: median spectrum for orientation, then each model's contribution profile (normalised to its max).
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(wn, np.median(Z, axis=0), lw=0.8, color="0.2")
    axes[0].set_ylabel("median SNV")
    panels = (
        ("species", "9 species (LOBO folds)"),
        ("soft_hard", "softwood vs hardwood"),
        ("date", "Fokienia date"),
    )
    for ax, (name, title) in zip(axes[1:], panels, strict=True):
        scale = prof[name].max()
        if name in band:
            ax.fill_between(
                wn, band[name][0] / scale, band[name][1] / scale, color="tab:blue", alpha=0.25, lw=0
            )
        ax.plot(wn, prof[name] / scale, lw=0.8, color="tab:blue")
        ax.set_ylabel("weight (rel.)")
        ax.set_title(title, fontsize=9, loc="left")
    for ax in axes:
        ax.axvspan(*REGIONS["silent"], color="tab:orange", alpha=0.12, lw=0)
        for b in WOOD_BANDS:
            ax.axvline(b, color="tab:green", lw=0.5, alpha=0.6)
        for b in NONBAND_PEAKS:
            ax.axvline(b, color="tab:red", lw=0.5, ls="--", alpha=0.6)
    axes[0].set_title(
        "orange = silent region (no wood bands) · green = known wood bands · red dashed = big peaks with no wood band",
        fontsize=8,
        loc="left",
    )
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out / "model_weights.png", dpi=150)

    config = run_config(
        "experiments/05_model_weights.py",
        window=WINDOW,
        regions=REGIONS,
        wood_bands=WOOD_BANDS,
        nonband_peaks=NONBAND_PEAKS,
        vendor_anchors=VENDOR_ANCHORS,
        near_cm1=NEAR,
        pca_components=20,
    )
    write_run(out, metrics, config)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
