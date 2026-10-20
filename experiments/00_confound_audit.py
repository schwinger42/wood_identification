"""00 — Confound audit. Run after any data change:  uv run python experiments/00_confound_audit.py

Question: how much of the apparent "species signal" is really session (date), moisture (O-H region)
or within-tree structure? Every number here uses a group split; nothing is spot-level random CV.

Writes results/<YYYYMMDD>_00_confound_audit/{metrics.json, inventory.csv, median_spectra.png}.
"""

from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline

from woodraman.io import REPO_ROOT, load_dataset
from woodraman.preprocess import SNV, Crop, spike_score_2nd_diff
from woodraman.splits import assert_no_leak, held_out_tree_folds, leave_one_group_out

REGIONS: dict[str, list[tuple[float, float]]] = {
    "fingerprint_400_1800": [(400, 1800)],
    # negative control: wood has no Raman bands here; any "species signal" here is not chemistry
    "silent_1850_2700": [(1850, 2700)],
    "CH_2800_3050": [(2800, 3050)],
    "OH_water_3050_3500": [(3050, 3500)],
    "full_400_3500": [(400, 3500)],
}

WITHIN_DAY_PAIRS = {
    "0925_Chamaecyparis_pair": (
        "20250925",
        ["Chamaecyparis formosensis", "Chamaecyparis obtusa var. formosana"],
    ),
    "0926_Swietenia_pair": ("20250926", ["Swietenia mahagoni", "Swietenia macrophylla"]),
    "0922_Cobtusa_vs_Fokienia": ("20250922", ["Chamaecyparis obtusa", "Fokienia hodginsii"]),
}


def model(wn: np.ndarray, region: str, n_components: int = 20):
    """PCA-LDA baseline. Crop/SNV are stateless; PCA and LDA are fitted inside each fold."""
    return make_pipeline(
        Crop(wn, REGIONS[region]),
        SNV(),
        PCA(n_components=n_components, svd_solver="full"),
        LinearDiscriminantAnalysis(),
    )


def grouped_cv(X, y, meta, wn, region, level="block_id") -> dict:
    """Leave-one-group-out; returns spot-level balanced acc and group-level (majority vote) acc."""
    pred = np.empty(len(y), dtype=object)
    for train, test in leave_one_group_out(meta, level):
        assert_no_leak(meta, train, test, level)
        n_comp = min(20, len(train) - 1)
        pred[test] = model(wn, region, n_comp).fit(X[train], y[train]).predict(X[test])
    df = pd.DataFrame({"g": meta[level].to_numpy(), "y": y, "p": pred})
    # majority vote over the test spectra of one (group, true label) unit, e.g. the 30 spectra of a block
    vote = df.groupby(["g", "y"]).p.agg(lambda s: s.mode().iloc[0]).reset_index()
    return {
        "split": f"leave-one-{level.removesuffix('_id')}-out",
        "spot_balanced_acc": round(float(balanced_accuracy_score(y, pred.astype(str))), 3),
        f"{level.removesuffix('_id')}_vote_acc": round(float((vote.y == vote.p).mean()), 3),
        "n_vote_units": int(vote.shape[0]),
        "chance": round(1 / len(np.unique(y)), 3),
    }


def main() -> None:
    X, wn, meta = load_dataset()
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_00_confound_audit"
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    # 1. Inventory -----------------------------------------------------------------------------
    inv = (
        meta.groupby(["species", "date"], dropna=False)
        .agg(
            power_mw=("power_mw", "first"),
            trees=("tree_id", "nunique"),
            blocks=("block_id", "nunique"),
            spectra=("path", "size"),
        )
        .reset_index()
    )
    inv.to_csv(out / "inventory.csv", index=False)
    print(inv.to_string(index=False))
    metrics["inventory"] = {
        "n_spectra": int(X.shape[0]),
        "n_points": int(X.shape[1]),
        "axis_cm1": [round(float(wn.min()), 1), round(float(wn.max()), 1)],
        "axis_step_cm1": [round(float(np.diff(wn).min()), 2), round(float(np.diff(wn).max()), 2)],
        "n_species": int(meta.species.nunique()),
        "n_trees": int(meta.tree_id.nunique()),
        "n_blocks": int(meta.block_id.nunique()),
        "species_with_gt1_tree": sorted(meta.groupby("species").tree_id.nunique().loc[lambda s: s > 1].index),
    }

    # 2. Data quality --------------------------------------------------------------------------
    nonpos = X <= 0
    level = np.median(X, axis=1)
    metrics["quality"] = {
        "frac_negative_values": round(float((X < 0).mean()), 4),
        # vendor fixed-anchor baseline? spectra pinned to <=0 in the same narrow windows
        "frac_spectra_nonpositive_in_955_980": round(
            float(nonpos[:, (wn > 955) & (wn < 980)].any(1).mean()), 3
        ),
        "frac_spectra_nonpositive_in_2810_2832": round(
            float(nonpos[:, (wn > 2810) & (wn < 2832)].any(1).mean()), 3
        ),
        "max_intensity": round(float(X.max()), 1),
        "median_of_spectrum_max": round(float(np.median(X.max(1))), 1),
        "n_sharp_spikes_2nd_diff": int((spike_score_2nd_diff(X) > 50).sum()),
        "whole_spectrum_outliers_median_gt_10x": meta.path[level > 10 * np.median(level)].tolist(),
    }

    # How much of each spectrum is one shared (non-species) shape? Silent-region structure = not chemistry.
    k = (wn >= 400) & (wn <= 3450)
    g = np.median(X[:, k], axis=0)
    gc = g - g.mean()
    Xc = X[:, k] - X[:, k].mean(1, keepdims=True)
    beta = Xc @ gc / (gc @ gc)
    r2 = 1 - ((Xc - beta[:, None] * gc) ** 2).sum(1) / (Xc**2).sum(1)
    sp_med = np.array(
        [np.median(X[(meta.species == s).to_numpy()][:, k], 0) for s in sorted(meta.species.unique())]
    )
    silent = (wn[k] >= 1850) & (wn[k] <= 2700)
    iu = np.triu_indices(len(sp_med), 1)
    metrics["shared_structure"] = {
        "median_R2_single_spectrum_vs_global_median": round(float(np.median(r2)), 3),
        "between_species_median_corr_400_3450": round(float(np.corrcoef(sp_med)[iu].mean()), 3),
        "between_species_median_corr_silent_1850_2700": round(
            float(np.corrcoef(sp_med[:, silent])[iu].mean()), 3
        ),
    }

    y_sp = meta.species.to_numpy()

    # 3. Which spectral region "predicts species"? (LOBO, within-tree, optimistic) -------------
    metrics["species_by_region_LOBO_optimistic"] = {
        r: grouped_cv(X, y_sp, meta, wn, r, "block_id") for r in REGIONS
    }

    # 4. Held-out tree: the only species-generalisation test the data allows -------------------
    held: dict = {}
    for r in ("fingerprint_400_1800", "silent_1850_2700", "OH_water_3050_3500", "full_400_3500"):
        held[r] = {}
        for tree, train, test in held_out_tree_folds(meta):
            assert_no_leak(meta, train, test, "tree_id")
            p = model(wn, r).fit(X[train], y_sp[train]).predict(X[test])
            vc = pd.Series(p).value_counts(normalize=True).round(3)
            held[r][tree] = {
                "recall": round(float((p == y_sp[test][0]).mean()), 3),
                "predicted_as": vc.head(3).to_dict(),
            }
    metrics["held_out_tree_9class"] = held

    # 5. Within-day species pairs: the only comparisons not confounded by session -------------
    pairs: dict = {}
    for name, (d, species) in WITHIN_DAY_PAIRS.items():
        m = (meta.date == d) & meta.species.isin(species)
        sub = meta[m].reset_index(drop=True)
        pairs[name] = {
            r: grouped_cv(X[m.to_numpy()], sub.species.to_numpy(), sub, wn, r, "block_id")
            for r in ("fingerprint_400_1800", "silent_1850_2700", "OH_water_3050_3500")
        }
        if sub.groupby("species").tree_id.nunique().max() > 1:
            pairs[name]["held_out_tree"] = {}
            for r in ("fingerprint_400_1800", "silent_1850_2700", "OH_water_3050_3500"):
                res = {}
                for tree, train, test in held_out_tree_folds(sub):
                    p = (
                        model(wn, r)
                        .fit(X[m.to_numpy()][train], sub.species.to_numpy()[train])
                        .predict(X[m.to_numpy()][test])
                    )
                    res[tree] = round(float((p == sub.species.iloc[test[0]]).mean()), 3)
                pairs[name]["held_out_tree"][r] = res
    metrics["within_day_pairs"] = pairs

    # 6. Session effect: same Fokienia blocks measured on 0922 and 0923 -----------------------
    fk = (meta.species == "Fokienia hodginsii").to_numpy()
    sub = meta[fk].reset_index(drop=True)
    metrics["session_effect_fokienia_date_LOBO"] = {
        r: grouped_cv(X[fk], sub.date.to_numpy(), sub, wn, r, "block_id")
        for r in ("fingerprint_400_1800", "silent_1850_2700", "OH_water_3050_3500")
    }

    # 7. Face (C/R/T) predictability: anisotropy / domain shift ------------------------------
    metrics["face_by_region_LOBO"] = {
        r: grouped_cv(X, meta.face.to_numpy(), meta, wn, r, "block_id")
        for r in ("fingerprint_400_1800", "OH_water_3050_3500")
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 8. Figure: median spectrum per species (SNV on 400-3500), water region shaded -----------
    keep = (wn >= 400) & (wn <= 3500)
    Z = SNV().transform(X[:, keep])
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sp in enumerate(sorted(meta.species.unique())):
        ax.plot(wn[keep], np.median(Z[y_sp == sp], axis=0) + 1.5 * i, lw=0.8, label=sp)
    ax.axvspan(3050, 3500, color="tab:blue", alpha=0.08, label="O-H / water 3050–3500")
    ax.axvspan(1600, 1680, color="tab:orange", alpha=0.08, label="H-O-H bend ~1640 / lignin")
    ax.set(
        xlabel="Raman shift (cm$^{-1}$)",
        ylabel="SNV intensity (offset)",
        title="Median spectrum per species (vendor-processed export)",
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "median_spectra.png", dpi=150)

    print(json.dumps({k: v for k, v in metrics.items() if k != "inventory"}, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
