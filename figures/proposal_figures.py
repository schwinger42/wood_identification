"""Proposal figures (zh-TW labels) from the latest committed experiment results.

    uv run python figures/proposal_figures.py

Reads results/<date>_00/02/03_*/metrics.json + CSVs (run those experiments first); recomputes nothing, so every
number in a figure is the number the experiment wrote. Refuses results produced by a dirty tree or a different
commit than HEAD. Every figure passes a layout check (no overlapping text, nothing outside the canvas) before saving.
Writes results/<YYYYMMDD>_proposal_figures/{fig1_ladder, fig2_signal_budget, fig3_same_day}.{png,svg} + config.json.
"""

from __future__ import annotations

import json
from datetime import date
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from woodraman.evaluate import _git, run_config
from woodraman.io import REPO_ROOT

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CHEM, CTRL, WATER, CHANCE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
ZH = {  # in measurement-date order
    "Acacia mangium": "馬占相思 9/18",
    "Machilus kusanoi": "大葉楠 9/19",
    "Chamaecyparis obtusa": "日本扁柏 9/22",
    "Fokienia hodginsii": "福建柏 9/22–23",
    "Acacia confusa": "相思樹 9/23",
    "Chamaecyparis formosensis": "紅檜 9/25",
    "Chamaecyparis obtusa var. formosana": "臺灣扁柏 9/25",
    "Swietenia mahagoni": "小葉桃花心木 9/26",
    "Swietenia macrophylla": "大葉桃花心木 9/26",
}


def latest(exp: str) -> Path:
    """Newest results/<date>_<exp> directory whose config says: this commit, clean tree."""
    dirs = sorted((REPO_ROOT / "results").glob(f"*_{exp}"))
    if not dirs:
        raise FileNotFoundError(f"No results for {exp}; run experiments/{exp}.py first")
    d = dirs[-1]
    cfg = json.loads((d / "config.json").read_text())
    head = _git("rev-parse", "HEAD")
    if cfg["git_dirty"] or cfg["git_commit"] != head:
        raise RuntimeError(
            f"{d.name} was produced by {cfg['git_commit'][:7]} (dirty={cfg['git_dirty']}), "
            f"HEAD is {head[:7]}: re-run experiments/{exp}.py"
        )
    return d


def layout_problems(fig) -> list[str]:
    """Overlapping text boxes or text outside the canvas, after drawing."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    undrawn = set()  # tick labels outside an axis' view interval exist as objects but are never drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    undrawn.update({id(tick.label1), id(tick.label2)})
    texts = [
        t
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip() and id(t) not in undrawn
    ]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    out = []
    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            out.append(f"outside canvas: {t.get_text()!r}")
    for (t1, b1), (t2, b2) in combinations(boxes, 2):
        dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if dx > 1 and dy > 1:
            out.append(f"overlap: {t1.get_text()!r} / {t2.get_text()!r}")
    return out


def save(fig, out: Path, name: str) -> None:
    problems = layout_problems(fig)
    if problems:
        raise AssertionError(f"{name}: " + "; ".join(problems))
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig1_ladder(d02: Path):
    lad = pd.read_csv(d02 / "ladder.csv").set_index(["region", "split"]).score
    sh = pd.read_csv(d02 / "soft_hard.csv").groupby(["split", "region"]).bacc.mean()
    perm = pd.read_csv(d02 / "soft_hard_permutation.csv").set_index("region").perm_p
    hot = "held-out tree (mean recall, 4 trees)"
    rnd = "spot-level random 5-fold"
    rows = [
        ("9 樹種｜隨機切分", lad["fingerprint", rnd], lad["silent", rnd], 1 / 9),
        (
            "9 樹種｜換試塊",
            lad["fingerprint", "leave-one-block-out"],
            lad["silent", "leave-one-block-out"],
            1 / 9,
        ),
        ("9 樹種｜換一棵樹（4 棵）", lad["fingerprint", hot], lad["silent", hot], 1 / 9),
        (
            "針葉 vs 闊葉｜換一棵樹（11 棵）",
            sh["leave-one-tree-out", "fingerprint"],
            sh["leave-one-tree-out", "silent"],
            0.5,
        ),
        (
            "針葉 vs 闊葉｜換一天",
            sh["leave-one-day-out", "fingerprint"],
            sh["leave-one-day-out", "silent"],
            0.5,
        ),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (_, chem, ctrl, ch) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([ch, ch], [y - 0.28, y + 0.28], color=CHANCE, lw=2)
        ax.plot([min(chem, ctrl), max(chem, ctrl)], [y, y], color="#cfcfca", lw=3, zorder=1)
        ax.scatter([chem], [y], s=90, color=CHEM, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter([ctrl], [y], s=90, color=CTRL, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(
            1.04,
            y,
            f"化學 {chem:.2f}　對照 {ctrl:.2f}　亂猜 {ch:.2f}",
            va="center",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("分數（各類答對率的平均）")
    ax.grid(axis="x", color="#ececea")
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=60, color=CHEM, label="化學波段 400–1800 cm⁻¹")
    ax.scatter([], [], s=60, color=CTRL, label="對照波段 1850–2700 cm⁻¹（無木材化學峰）")
    ax.plot([], [], color=CHANCE, lw=2, label="亂猜")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.01,
        0.01,
        f"針葉 vs 闊葉兩列為 4 種分類器平均；換一天的亂貼標籤檢定 p = {perm['fingerprint']:.2f}"
        "（化學波段）。資料：experiments/02。",
        fontsize=8,
        color="#555",
    )
    fig.subplots_adjust(left=0.24, right=0.74, top=0.86, bottom=0.2)
    return fig


def fig2_budget(d03: Path):
    raw = pd.read_csv(d03 / "signal_budget.csv")
    b = raw.pivot(index="effect", columns="region", values="value")
    n = raw.groupby("effect").n_pairs.first()
    order = [
        ("tree vs tree (same species, same day)", "同一樹種、不同棵（同一天）"),
        ("spot noise (single spectrum)", "單條光譜本身的雜訊"),
        ("species vs species (same day)", "不同樹種（同一天）"),
        ("block vs block (same tree, same day)", "同一棵、不同試塊"),
        ("day vs day (same wood blocks)", "同一試塊、換一天量"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    h = 0.36
    for i, (key, _) in enumerate(order):
        y = len(order) - 1 - i
        for dy, col, val in ((h / 2, CHEM, b.loc[key, "fingerprint"]), (-h / 2, CTRL, b.loc[key, "silent"])):
            ax.barh(y + dy, val, height=h * 0.9, color=col)
            ax.text(val + 0.006, y + dy, f"{val:.2f}", va="center", fontsize=8.5)
    labels = [f"{lab} · {n[key]} 組" for key, lab in order][::-1]
    ax.set_yticks(range(len(order)), labels, fontsize=10)
    ax.get_yticklabels()[2].set_fontweight("bold")  # the species row: the signal we need
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("兩組平均光譜的差異（均方根，單位：一條光譜自身起伏的標準差；已扣抽樣誤差）")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=CHEM, label="化學波段"), Patch(color=CTRL, label="對照波段（無木材化學峰）")]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9
    )
    fig.text(
        0.01, 0.01, "組數少（最少 2 組），只能當量級看。資料：experiments/03。", fontsize=8, color="#555"
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.2)
    return fig


def fig3_same_day(d00: Path):
    m = json.loads((d00 / "metrics.json").read_text())
    kinds = m["species_pairs_vs_day"]
    names = list(ZH)
    idx = {s: i for i, s in enumerate(names)}
    M = np.zeros((len(names), len(names)))  # 0 blank, 1 aliased, 2 bridged, 3 same day
    for k, v in (("aliased_with_day", 1), ("bridged", 2), ("same_day", 3)):
        for a, b in kinds[k]:
            i, j = sorted((idx[a], idx[b]))
            M[i, j] = v
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    cols = {
        1: "#e3e3df",
        2: "#eda100",
        3: "#2e5c4f",
    }  # amber, not CHEM: blue means "chemistry band" on the right
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ax1.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=cols[int(M[i, j])]))
    ax1.set_xlim(-0.6, len(names) - 0.4)
    ax1.set_ylim(len(names) - 0.4, -0.6)
    ax1.set_xticks(range(len(names)), [ZH[s] for s in names], rotation=90, fontsize=8.5)
    ax1.set_yticks(range(len(names)), [ZH[s] for s in names], fontsize=8.5)
    ax1.set_aspect("equal")
    ax1.spines[:].set_visible(False)
    ax1.tick_params(length=0)
    c = kinds["counts"]
    ax1.set_title(
        f"36 組樹種配對：同一天 {c['same_day']} 組、經福建柏間接 {c['bridged']} 組、"
        f"與量測日綁死 {c['aliased_with_day']} 組",
        fontsize=9.5,
        loc="left",
    )
    for col, lab in (
        (cols[3], "同一天量，可直接比"),
        (cols[2], "只能經福建柏間接比"),
        (cols[1], "與量測日綁死"),
    ):
        ax1.scatter([], [], marker="s", s=60, color=col, label=lab)
    ax1.legend(loc="lower left", frameon=False, fontsize=8.5)

    pairs = [
        ("0925_Chamaecyparis_pair", "紅檜 vs\n臺灣扁柏"),
        ("0926_Swietenia_pair", "小葉 vs 大葉\n桃花心木"),
        ("0922_Cobtusa_vs_Fokienia", "日本扁柏 vs\n福建柏"),
        ("0923_Aconfusa_vs_Fokienia", "相思樹 vs\n福建柏"),
    ]
    regs = (
        ("fingerprint_400_1800", "化學波段", CHEM),
        ("silent_1850_2700", "對照波段", CTRL),
        ("OH_water_3050_3500", "水波段", WATER),
    )
    w = 0.26
    for i, (key, _) in enumerate(pairs):
        for k, (r, lab, col) in enumerate(regs):
            v = m["within_day_pairs"][key][r]["spot_balanced_acc"]
            ax2.bar(i + (k - 1) * w, v, width=w * 0.92, color=col)
            ax2.text(i + (k - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax2.axhline(0.5, color=CHANCE, lw=1.5, ls="--")
    ax2.set_xticks(range(len(pairs)), [p[1] for p in pairs], fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("分數（換試塊測試）")
    ax2.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=col, label=lab) for _, lab, col in regs]
    handles.append(Line2D([], [], color=CHANCE, lw=1.5, ls="--", label="亂猜 0.5"))
    ax2.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9
    )
    fig.text(
        0.01,
        0.01,
        "右圖只含同一天量的 4 組；只有紅檜 vs 臺灣扁柏是化學波段明顯領先。各樹種僅 1–2 棵，"
        "表示「這幾棵樹分得開」，不是樹種層級的辨識。資料：experiments/00。",
        fontsize=8,
        color="#555",
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.27, wspace=0.28)
    return fig


def main() -> None:
    d00, d02, d03 = (
        latest("00_confound_audit"),
        latest("02_generalisation_ladder"),
        latest("03_signal_budget"),
    )
    out = REPO_ROOT / "results" / f"{date.today():%Y%m%d}_proposal_figures"
    out.mkdir(parents=True, exist_ok=True)
    save(fig1_ladder(d02), out, "fig1_ladder")
    save(fig2_budget(d03), out, "fig2_signal_budget")
    save(fig3_same_day(d00), out, "fig3_same_day")
    cfg = run_config("figures/proposal_figures.py", sources=[d.name for d in (d00, d02, d03)])
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
