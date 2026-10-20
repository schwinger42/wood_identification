# CLAUDE.md — Raman wood species identification (v1)

## Purpose and audience
- PoC with Prof. Tsao (NCHU): identify Taiwanese timber species from Raman spectra with ML.
- Goal of the PoC: results that survive peer review and support an NSTC funding proposal.
  An honest 70% under a tree-level split is worth more than 99% under a leaky split.
- Talk to me in Traditional Chinese with English technical terms. Be direct; flag weak
  methodology even when I didn't ask. If I start re-planning instead of running the next experiment, say so.

## Commands
- Environment: `uv sync` (Python >= 3.12, lockfile `uv.lock`). Add deps with `uv add <pkg>`.
- Run anything: `uv run python <script>`; tests: `uv run pytest`; lint: `uv run ruff check .`;
  notebooks: `uv run jupyter lab`.
- Confound audit (run after ANY data or pipeline change): `uv run python experiments/00_confound_audit.py` (~1 min)
- Method sweep, 36 preprocessing × classifier combos vs negative controls: `uv run python experiments/01_method_sweep.py` (~40 min on 4 cores)
- Generalisation ladder + softwood/hardwood unseen-day test + band-feature controls: `uv run python experiments/02_generalisation_ladder.py` (~3 min)
- Legacy v0 pipeline + Streamlit demo: git tag `v0-legacy`. To run the old demo without polluting this tree:
  `git worktree add ../wood_v0 v0-legacy`. Never copy v0 code or "proven parameters" back into `src/`.

## Data facts (verified 2026-10-21 by experiments/00_confound_audit.py)
- `dataset/raw/<species>/**/*.txt`, 1050 spectra, `;`-separated, CRLF, 3 columns:
  Raman shift (cm-1), intensity, all-zero column. 2005 points, 98.6–3509.8 cm-1,
  non-uniform spacing (1.01–2.59 cm-1), identical axis in every file.
- Exported data looks vendor-processed: ~3% negative values; 78% / 81% of spectra are <= 0 inside
  955–980 / 2810–2832 cm-1 (spectrum minima cluster there) → likely fixed-anchor baseline subtraction by the
  instrument software. Raw export is NOT in the repo; ask Prof. Tsao.
- **The dominant structure is not wood chemistry.** One shared shape explains a median 74% of each
  spectrum's variance; between-species median correlation is 0.84 overall and 0.83 in the Raman-silent region
  1850–2700 cm-1 (no wood bands there). The shape is stable across days (per-date medians correlate
  0.94–0.99 with the global median; no wavenumber drift > 0.5 cm-1) → instrument-fixed pattern.
- Real wood bands ARE present but small: peaks at ~1327, ~1457, ~1661, ~1107 and ~2957 (C-H, prominence 683)
  match expected bands, yet bumps of the same size sit in the silent region (2259/2490/2628: 616–625) and the
  biggest peaks (1016, 648, 832) match no wood band. 1600 (lignin, normally the strongest band) is a DIP.
  The vendor anchor at 2810–2832 sits on the low edge of the C-H band and may distort it.
- The pattern's overall amplitude alone does NOT predict species (LOBO 0.15, chance 0.11); the shortcut is in
  shape, not scale. No consistent wavenumber-axis offset vs expected band positions.
- Hypothesis to test with raw exports + an intensity standard: lignin fluorescence × uncorrected instrument
  response (filter ripple / pixel gain) survives vendor baseline subtraction as a fixed pattern.
- 3 whole-spectrum outliers (median level >10× typical, up to 52.6k counts): `C. obtusa var. formosana 2-4-C-8`,
  `Fokienia 1-T-8` (0922), `S. macrophylla 2-1-T-6`. Not cosmic rays (no sharp 2nd-difference spikes found
  anywhere). Flag them; never silently clip.
- Hierarchy: species → tree → block (cube) → face (C/R/T) → spot. 30 spectra/block (Fokienia on 0923: 15).
- Three filename variants, parsed by `woodraman.io.parse_filename`:
  `24mW-<date>-<sp>-<tree>-<block>-<face>-<spot>`, `24mW-<date>-<sp>-<block>-<face>-<spot>`,
  `<date>-<sp>-<block>-<face>-<spot>` (laser power NOT recorded; legacy code assumed 100 mW).
- 9 species, 11 trees, 34 blocks. Only C. obtusa var. formosana and S. macrophylla have 2 trees.
  The statistical unit is the TREE, not the spectrum.
- Folder typos are kept on disk and mapped in code (`SPECIES_CANONICAL`): `Swietenia marophylla` /
  `marcrophylla` = S. macrophylla; `Chamaecyparis obyusa` = ASSUMED C. obtusa (Japanese hinoki) — confirm.

## Known confounds (each species was measured on its own day)
| date | species | power |
|---|---|---|
| 0918 | Acacia mangium | unrecorded |
| 0919 | Machilus kusanoi | unrecorded |
| 0922 | C. obtusa (JP), Fokienia (B1, B3) | unrecorded |
| 0923 | Acacia confusa, Fokienia (B1, B3 again, spots 11–15) | unrecorded |
| 0925 | C. formosensis, C. obtusa var. formosana | 24 mW |
| 0926 | S. mahagoni, S. macrophylla | 24 mW |
- Session effect: on the same two Fokienia blocks, date is predictable on the held-out block
  (balanced acc 0.93 fingerprint, 0.98 silent region; only 2 blocks, so weak evidence but consistent).
- Silent region 1850–2700 predicts species as well as the fingerprint (LOBO 0.74 vs 0.73, chance 0.11) and
  better on held-out Swietenia trees (0.84/0.84 vs 0.62/0.42). O-H region alone: LOBO 0.69.
- Held-out-tree recall (9-class, fingerprint): C. obtusa var. formosana 0.33/0.33, S. macrophylla 0.62/0.42.
- Inflation ladder (9-class PCA-LDA, fingerprint / silent): spot-level random 5-fold 0.91 / 0.93 →
  LOBO 0.73 / 0.74 → held-out tree 0.43 / 0.60. Quote spot-level only as this demonstration.
- The silent-region shortcut lives in NARROW structure (<120 cm-1 scale: LOBO 0.73, date 0.98), not in the
  broad background (LOBO 0.52) → baseline correction or derivatives cannot remove it.
- Method sweep (experiments/01, 6 preprocessings × 6 classifiers incl. RF/MLP, untuned): best within-day
  held-out-tree recall 0.71 (SG1+SNV+RF) vs its silent control 0.69; max fingerprint−silent margin 0.10.
  C. obtusa var. formosana tree 2 is never recognised (0.12–0.47 in all 36) — it looks like C. formosensis.
  EPO from Fokienia session pairs removes the date signal but does not improve species generalisation.
  Picking the best of 36 on 4 test trees is itself selection bias. With 4 held-out trees, even 4/4 has a
  95% CI of [0.40, 1.00]: no method can make a species-level claim from this data.
- **Unseen-day test (experiments/02):** softwood vs hardwood (chemically the easiest split), leave-one-DAY-out:
  balanced acc fingerprint 0.37, silent 0.39, C-H 0.23, O-H 0.18 — BELOW chance 0.5 — vs 0.62–0.69 with
  leave-one-tree-out. Whatever the models learn does not survive a new measurement day.
- Known-band features (13 wood bands) do no better than the same features at random off-band fingerprint
  positions (experiments/02). 0925 pair: fingerprint beats silent in 9/10 methods but by only ~0.06 on average.
- Within-day pairs are the only species comparisons not confounded by session. Best evidence of chemistry:
  0925 Chamaecyparis pair, LOBO fingerprint 0.73 > silent 0.58 > O-H 0.47 (chance 0.5) — but held-out tree
  0.86 / 0.39. In the 0926 Swietenia pair, O-H (0.92) and silent (0.81) match or beat fingerprint (0.83).

## Evaluation contract (non-negotiable)
1. Never report spot-level random K-fold. Every result states its split explicitly.
2. Primary: leave-one-tree-out wherever a species has >1 tree. Secondary: leave-one-block-out
   (LOBO), labelled "within-tree, optimistic". Grouping uses `block_id`/`tree_id`, never `spot`.
3. All fitted preprocessing (scaler, PCA, EMSC reference, feature selection, augmentation
   statistics) is fitted inside the training fold — use sklearn Pipelines (`woodraman.preprocess`).
4. Report balanced accuracy, macro-F1, per-class recall and the confusion matrix. Also report
   block-level accuracy (majority vote over a block's spectra) as the operational metric.
5. Negative controls for every new pipeline, reported next to the main number:
   (a) within-day pair accuracy, (b) O-H-region-only, (c) **silent-region-only (1850–2700)**,
   (d) date prediction on Fokienia. If (b) or (c) is close to the main result, the pipeline is not
   measuring wood chemistry — say so in the write-up.
6. Hyperparameters are tuned with nested group CV. No tuning on held-out trees.
7. Compare every DL model against PCA-LDA / PLS-DA / linear SVM baselines under the same split.
8. Seeds fixed; each run writes `results/<date>_<exp>/metrics.json` + config. No result
   goes into a slide or paper unless it is reproducible from a committed script.

## Code layout
- `src/woodraman/`: `io.py` (parsing + metadata), `preprocess.py` (Crop, SNV, SGDerivative, ASPLSBaseline,
  NuisanceProjector/EPO), `splits.py` (group splitters + `assert_no_leak`), `models.py` (PLS-DA + fixed
  baseline classifiers). To add: EMSC, `evaluate.py`.
- `experiments/NN_<name>.py`: thin scripts importing `woodraman`; one question per script.
- `notebooks/`: exploration only; move anything reusable into `src/`.
- `results/`, `models/`: generated and git-ignored.
- `docs/`: `literature-2026-10.md`, `measurement-protocol-v1.md`.
- No absolute paths; resolve from `woodraman.io.REPO_ROOT`. Type hints, small pure functions,
  pytest for parsing and splitters (`tests/`).

## Spectroscopy notes
- Wood Raman bands, checked 2026-10-21 against sources (Agarwal 2019 Molecules 24:1659; Yamauchi et al. 2005
  J Wood Sci 51:498; Vardaki et al. 2024 RSC Chem Biol 5:273; Masson et al. 2018 Analyst):
  380 cellulose; 370 + 417–531 S-lignin (hardwoods); 1095/1100 cellulose; 1255–1290 G-lignin; 1320–1338 S-lignin;
  1380 and 1460/1480 cellulose (CH2/CH deformation); 1540–1700 lignin, incl. 1600 (conjugated aromatic) and
  1655–1660 (C=C, C=O); 2800–2918 carbohydrate C-H, 2940 S-lignin, 2800–3030 all C-H;
  1800–2800 "silent" (only triple bonds / C-D; absent from biological molecules); ~3035–3680 O-H (water).
  Water: broad O-H stretch is the main band; weak H-O-H bend ~1640 overlaps lignin bands.
  Water is a weak Raman scatterer; at 532 nm the bigger moisture effect is on lignin FLUORESCENCE
  (moisture quenches it, Peters & Rapp 2024), i.e. on the background, across the whole window.
- SERDS (two shifted lasers) removes fluorescence/fixed-pattern background, NOT water Raman bands.
- 532 nm can alter lignin bands (Prats-Mateu 2018): laser dose must be fixed and recorded.
- Preprocessing order to test, not assume: cosmic-ray check → crop → baseline (asPLS/arPLS/SNIP) →
  normalisation (SNV/vector/area) → optional EMSC/EPO with water as interferent. Report sensitivity.

## Research direction (details: docs/literature-2026-10.md)
- Peer-reviewed Raman + ML wood *species* work is thin (Sowoidnich 2023, 830 nm SERDS, 5 species,
  PLS-DA); none found for 2024–2026; no Raman-wood paper models moisture; no public Raman wood DB.
- Leakage is the field's main flaw: image-level CV 0.95 → leave-trees-out 0.22–0.30
  (Rosa da Silva et al. 2022, Plant Methods). Cite it when defending tree-level splits.
- Current bottleneck is ACQUISITION (see audit), not the model. CONSTRAINT: no new measurements during the
  PoC — the 1050 spectra are fixed. Free asks that are not new measurements: re-export the SAME measurements
  raw (before vendor baseline) if the instrument software kept them; measurement order/time stamps;
  instrument parameters. `docs/measurement-protocol-v1.md` is the data plan for the funding proposal.
- At this n, DL is not expected to beat PCA-LDA/PLS-DA/SVM. DL work = learning curves +
  augmentation + optional pretraining, framed as a controlled comparison, not the headline.
- Candidate contributions: (1) hierarchical split audit + variance decomposition + negative controls;
  (2) moisture-robust Raman via paired drying series; (3) face (C/R/T) as domain shift;
  (4) congeneric pairs with forensic value (C. formosensis vs C. obtusa var. formosana; Swietenia, CITES II).
- Any species-level claim needs ≥3 trees/species (target 5). Until then, phrase results as
  "discriminating these individual trees" and focus on protocol + feasibility.

## Open questions for Prof. Tsao (block decisions; full list in docs/measurement-protocol-v1.md)
- Instrument model, grating, objective, integration time, accumulations; power for un-prefixed files.
- Raw export (before vendor baseline) + dark + intensity standard possible?
- Moisture state of samples on each date; what "the water signal" looks like on screen.
- Is `Chamaecyparis obyusa` Japanese hinoki? Access to 785 nm or 1064 nm FT-Raman?

## Legacy (v0, Sep–Oct 2025) — do not trust its claims
- Full-range SG smoothing + morphological baseline + min-shift + L2 → soft-voting XGB/RF/SVM.
  Reported leave-one-cube-out 0.855 ± 0.185; held-out-tree test 0.56 on 2 species. Given the
  silent-region result, treat these numbers as session/instrument separability, not species ID.
  v0's `reference/` and `.junie/` were AI-generated drafts with unverified "proven parameters".
