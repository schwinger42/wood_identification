"""03 — Signal budget: how big is each source of spectral difference?  uv run python experiments/03_signal_budget.py

Every effect is the RMS difference between two group-mean spectra (SNV, per Raman-shift point), debiased for the
sampling noise of the means: d^2 = mean_k[(m1-m2)^2 - s1^2/n1 - s2^2/n2]. Same units for every row, so they compare.
  spot noise           : one spectrum vs its block mean (what a single measurement scatters by)
  block vs block       : two blocks of the same tree, same day
  tree vs tree         : two trees of the same species, same day
  species vs species   : two species measured on the same day (the signal we want)
  day vs day           : the SAME Fokienia blocks measured on 0922 and on 0923 (pure measurement-day effect)
Run on the fingerprint region (wood chemistry) and the silent region (no wood chemistry: any "effect" there is
instrument/background, i.e. the floor that chemistry has to rise above).
Writes results/<YYYYMMDD>_03_signal_budget/{signal_budget.csv, metrics.json, config.json}.
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
]


def rms_diff(A: np.ndarray, B: np.ndarray) -> float:
    d2 = (A.mean(0) - B.mean(0)) ** 2 - A.var(0, ddof=1) / len(A) - B.var(0, ddof=1) / len(B)
    return float(np.sqrt(max(d2.mean(), 0.0)))


def budget(Z: np.ndarray, meta: pd.DataFrame) -> list[dict]:
    rows = []
    # spot noise: RMS of spectra around their (block, date) mean
    res = []
    for _, g in meta.groupby(["block_id", "date"]):
        Zg = Z[g.index]
        res.append(((Zg - Zg.mean(0)) ** 2).mean())
    rows.append(
        {"effect": "spot noise (single spectrum)", "value": float(np.sqrt(np.mean(res))), "n_pairs": len(res)}
    )
    # block vs block, same tree, same day
    vals = []
    for (_, d), g in meta.groupby(["tree_id", "date"]):
        for b1, b2 in combinations(sorted(g.block_id.unique()), 2):
            vals.append(rms_diff(Z[g.index[g.block_id == b1]], Z[g.index[g.block_id == b2]]))
    rows.append(
        {
            "effect": "block vs block (same tree, same day)",
            "value": float(np.mean(vals)),
            "n_pairs": len(vals),
        }
    )
    # tree vs tree, same species, same day
    vals = []
    for (_, d), g in meta.groupby(["species", "date"]):
        for t1, t2 in combinations(sorted(g.tree_id.unique()), 2):
            vals.append(rms_diff(Z[g.index[g.tree_id == t1]], Z[g.index[g.tree_id == t2]]))
    rows.append(
        {
            "effect": "tree vs tree (same species, same day)",
            "value": float(np.mean(vals)),
            "n_pairs": len(vals),
        }
    )
    # species vs species, same day
    vals = []
    for d, s1, s2 in SAME_DAY_SPECIES:
        g = meta[meta.date == d]
        vals.append(rms_diff(Z[g.index[g.species == s1]], Z[g.index[g.species == s2]]))
    rows.append(
        {"effect": "species vs species (same day)", "value": float(np.mean(vals)), "n_pairs": len(vals)}
    )
    # day vs day, same wood (Fokienia blocks on 0922 and 0923)
    vals = []
    g = meta[meta.species == "Fokienia hodginsii"]
    for _, gb in g.groupby("block_id"):
        if gb.date.nunique() == 2:
            d1, d2 = sorted(gb.date.unique())
            vals.append(rms_diff(Z[gb.index[gb.date == d1]], Z[gb.index[gb.date == d2]]))
    rows.append(
        {"effect": "day vs day (same wood blocks)", "value": float(np.mean(vals)), "n_pairs": len(vals)}
    )
    return rows


def main() -> None:
    X, wn, meta = load_dataset()
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_03_signal_budget"
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for region, win in REGIONS.items():
        Z = SNV().transform(Crop(wn, win).fit(X).transform(X))
        for r in budget(Z, meta):
            allrows.append({"region": region, **r})
    df = pd.DataFrame(allrows)
    df.to_csv(out / "signal_budget.csv", index=False)
    config = run_config("experiments/03_signal_budget.py", regions=REGIONS, same_day_species=SAME_DAY_SPECIES)
    write_run(out, {"signal_budget": df.to_dict(orient="records")}, config)
    print(df.pivot(index="effect", columns="region", values="value").round(3).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
