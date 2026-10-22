"""03 — Signal budget: how big is each source of spectral difference?  uv run python experiments/03_signal_budget.py

Every effect is the RMS difference between two group-mean spectra (SNV, per Raman-shift point). Spectra from the
same block or face are NOT independent, so each group mean is built from UNIT means (unit = block, or face for
comparisons inside one block) and debiased with the between-unit variance:
d^2 = mean_k[(m1-m2)^2 - s1^2/n1 - s2^2/n2], with s^2 and n over units. Same units for every row, so rows compare.
  spot noise           : one spectrum vs its (block, day, face) mean — what a single measurement scatters by
  block vs block       : two blocks of the same tree, same day            (unit: face)
  tree vs tree         : two trees of the same species, same day          (unit: block)
  species vs species   : two species measured on the same day             (unit: block) — the signal we want
  day vs day           : the SAME Fokienia blocks on 0922 and on 0923     (unit: face) — pure measurement-day effect
Every pair is also written separately: the means hide a wide spread, and n is tiny (2–4 pairs for most rows).
Run on the fingerprint region (wood chemistry) and the silent region (no wood chemistry: any "effect" there is
instrument/background, i.e. the floor that chemistry has to rise above).
Revised 2026-10-22 after review: added the 0923 pair, spot noise within face, unit-level debiasing.
Writes results/<YYYYMMDD>_03_signal_budget/{signal_budget.csv, pairs.csv, metrics.json, config.json}.
"""

from __future__ import annotations

from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from woodraman.evaluate import run_config, write_run
from woodraman.io import REPO_ROOT, load_dataset
from woodraman.preprocess import SNV, Crop

REGIONS = {"fingerprint": [(400, 1800)], "silent": [(1850, 2700)]}
SAME_DAY_SPECIES = [
    ("20250925", "Chamaecyparis formosensis", "Chamaecyparis obtusa var. formosana"),
    ("20250926", "Swietenia mahagoni", "Swietenia macrophylla"),
    ("20250922", "Chamaecyparis obtusa", "Fokienia hodginsii"),
    ("20250923", "Acacia confusa", "Fokienia hodginsii"),
]
DEBIAS_UNIT = {
    "block vs block": "face",
    "tree vs tree": "block",
    "species vs species": "block",
    "day vs day": "face",
}


def unit_means(Z: np.ndarray, meta: pd.DataFrame, rows, unit: str) -> np.ndarray:
    """Mean spectrum of each unit (each block, or each face) among `rows`."""
    col = "block_id" if unit == "block" else unit
    g = meta.loc[rows]
    return np.array([Z[gg.index].mean(0) for _, gg in g.groupby(col)])


def rms_diff(A: np.ndarray, B: np.ndarray) -> float:
    """Debiased RMS difference of two group means; rows of A and B are unit means."""
    d2 = (A.mean(0) - B.mean(0)) ** 2 - A.var(0, ddof=1) / len(A) - B.var(0, ddof=1) / len(B)
    return float(np.sqrt(max(d2.mean(), 0.0)))


def budget(Z: np.ndarray, meta: pd.DataFrame) -> list[dict]:
    """One row per compared pair (spot noise: one row for the pooled within-face scatter)."""
    rows = []
    res = [
        ((Z[g.index] - Z[g.index].mean(0)) ** 2).mean() for _, g in meta.groupby(["block_id", "date", "face"])
    ]
    rows.append({"effect": "spot noise", "pair": f"{len(res)} faces", "value": float(np.sqrt(np.mean(res)))})
    for (_, d), g in meta.groupby(["tree_id", "date"]):
        for b1, b2 in combinations(sorted(g.block_id.unique()), 2):
            A = unit_means(Z, meta, g.index[g.block_id == b1], DEBIAS_UNIT["block vs block"])
            B = unit_means(Z, meta, g.index[g.block_id == b2], DEBIAS_UNIT["block vs block"])
            rows.append({"effect": "block vs block", "pair": f"{b1} / {b2}", "value": rms_diff(A, B)})
    for (sp, d), g in meta.groupby(["species", "date"]):
        for t1, t2 in combinations(sorted(g.tree_id.unique()), 2):
            A = unit_means(Z, meta, g.index[g.tree_id == t1], DEBIAS_UNIT["tree vs tree"])
            B = unit_means(Z, meta, g.index[g.tree_id == t2], DEBIAS_UNIT["tree vs tree"])
            rows.append({"effect": "tree vs tree", "pair": f"{sp} {d}", "value": rms_diff(A, B)})
    for d, s1, s2 in SAME_DAY_SPECIES:
        g = meta[meta.date == d]
        A = unit_means(Z, meta, g.index[g.species == s1], DEBIAS_UNIT["species vs species"])
        B = unit_means(Z, meta, g.index[g.species == s2], DEBIAS_UNIT["species vs species"])
        rows.append({"effect": "species vs species", "pair": f"{s1} / {s2} {d}", "value": rms_diff(A, B)})
    for b, gb in meta[meta.species == "Fokienia hodginsii"].groupby("block_id"):
        if gb.date.nunique() == 2:
            d1, d2 = sorted(gb.date.unique())
            A = unit_means(Z, meta, gb.index[gb.date == d1], DEBIAS_UNIT["day vs day"])
            B = unit_means(Z, meta, gb.index[gb.date == d2], DEBIAS_UNIT["day vs day"])
            rows.append({"effect": "day vs day", "pair": f"{b} {d1}/{d2}", "value": rms_diff(A, B)})
    return rows


def main() -> None:
    X, wn, meta = load_dataset()
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_03_signal_budget"
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for region, win in REGIONS.items():
        Z = SNV().transform(Crop(wn, win).fit(X).transform(X))
        pairs += [{"region": region, **r} for r in budget(Z, meta)]
    pr = pd.DataFrame(pairs)
    pr.to_csv(out / "pairs.csv", index=False)
    df = (
        pr.groupby(["region", "effect"])
        .value.agg(value="mean", n_pairs="size", min="min", max="max")
        .reset_index()
    )
    df.to_csv(out / "signal_budget.csv", index=False)

    # Same genus, same day: the two trees of one species vs the two species of that genus (fingerprint)
    fp = pr[pr.region == "fingerprint"].set_index("pair").value
    same_genus = {
        "Chamaecyparis 0925": {
            "tree_vs_tree": float(fp["Chamaecyparis obtusa var. formosana 20250925"]),
            "species_vs_species": float(
                fp["Chamaecyparis formosensis / Chamaecyparis obtusa var. formosana 20250925"]
            ),
        },
        "Swietenia 0926": {
            "tree_vs_tree": float(fp["Swietenia macrophylla 20250926"]),
            "species_vs_species": float(fp["Swietenia mahagoni / Swietenia macrophylla 20250926"]),
        },
    }
    config = run_config(
        "experiments/03_signal_budget.py",
        regions=REGIONS,
        same_day_species=SAME_DAY_SPECIES,
        debias_unit=DEBIAS_UNIT,
    )
    metrics = {
        "signal_budget": df.to_dict(orient="records"),
        "pairs": pr.to_dict(orient="records"),
        "same_genus_fingerprint": same_genus,
    }
    write_run(out, metrics, config)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False), "\n")
    print(pr[pr.effect != "spot noise"].round(3).to_string(index=False), "\n")
    print(
        "same genus, same day (fingerprint):",
        {k: {a: round(b, 3) for a, b in v.items()} for k, v in same_genus.items()},
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
