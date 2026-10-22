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
- Confound audit (run after ANY data or pipeline change): `uv run python experiments/00_confound_audit.py` (~45 s)
- Method sweep, 36 preprocessing × classifier combos vs negative controls: `uv run python experiments/01_method_sweep.py` (~2 min)
- Generalisation ladder + softwood/hardwood unseen-day test with permutation p + band-feature controls:
  `uv run python experiments/02_generalisation_ladder.py` (~3 min)
- Signal budget (how big each source of spectral difference is): `uv run python experiments/03_signal_budget.py` (~1 s)
- Shared instrument pattern (peaks, day stability, amplitude-only, broad vs narrow scale):
  `uv run python experiments/04_shared_pattern.py` (~15 s)
- Where linear models put their weight (replaces SHAP/LIME for linear models; paper figure):
  `uv run python experiments/05_model_weights.py` (~35 s)
- Proposal figures (zh-TW) from the latest results of 00/02/03; refuses stale or dirty results, checks layout:
  `uv run python figures/proposal_figures.py` (~10 s) → `results/<date>_proposal_figures/`
- Timings are with `OMP_NUM_THREADS=8 LOKY_MAX_CPU_COUNT=8` (2026-10-22). Set these when running scripts in parallel:
  uncapped BLAS threads in 4 parallel runs drove load to ~96 on 32 cores and made a 45 s script take 15+ min.
- Every script writes `results/<YYYYMMDD>_<exp>/metrics.json` + `config.json` (commit, dirty flag, versions,
  parameters) via `woodraman.evaluate.write_run`, next to its CSVs.
- Legacy v0 pipeline + Streamlit demo: git tag `v0-legacy`. To run the old demo without polluting this tree:
  `git worktree add ../wood_v0 v0-legacy`. Never copy v0 code or "proven parameters" back into `src/`.

## Data facts (each bullet names its source script, `[00]` = experiments/00_…; re-run 2026-10-22)
Rule: a number enters this file only with the script that reproduces it. No source = say UNVERIFIED.
- `dataset/raw/<species>/**/*.txt`, 1050 spectra, `;`-separated, CRLF, 3 columns:
  Raman shift (cm-1), intensity, all-zero column. 2005 points, 98.6–3509.8 cm-1,
  non-uniform spacing (1.01–2.59 cm-1), identical axis in every file. [00, `woodraman.io`]
- Hypothesis to test with raw exports + an intensity standard: lignin fluorescence × uncorrected instrument
  response (filter ripple / pixel gain) survives vendor baseline subtraction as a fixed pattern.
- 3 whole-spectrum outliers (median level >10× typical, up to 52.6k counts): `C. obtusa var. formosana 2-4-C-8`,
  `Fokienia 1-T-8` (0922), `S. macrophylla 2-1-T-6`. Not cosmic rays (no sharp 2nd-difference spikes found
  anywhere). Flag them; never silently clip. [00]
- Hierarchy: species → tree → block (cube) → face (C/R/T) → spot. 30 spectra/block (Fokienia on 0923: 15). [00]
- Three filename variants, parsed by `woodraman.io.parse_filename`:
  `24mW-<date>-<sp>-<tree>-<block>-<face>-<spot>`, `24mW-<date>-<sp>-<block>-<face>-<spot>`,
  `<date>-<sp>-<block>-<face>-<spot>` (laser power NOT recorded; legacy code assumed 100 mW).
- 9 species, 11 trees, 34 blocks. Only C. obtusa var. formosana and S. macrophylla have 2 trees.
  The statistical unit is the TREE, not the spectrum.
- Folder typos are kept on disk and mapped in code (`SPECIES_CANONICAL`): `Swietenia marophylla` /
  `marcrophylla` = S. macrophylla; `Chamaecyparis obyusa` = ASSUMED C. obtusa (Japanese hinoki) — confirm.
- `Swietenia marophylla/test/` and `Chamaecyparis obtusa var. formosana/test/` (90 files each) are simply
  TREE 2 of those species, present since the first upload (2025-10-12). The name is not a curated test set;
  the loader ignores folder names (rglob + filename parsing). Do not move the files.

## What the data show — conclusions that change how to work (numbers + sources: docs/research-status-2026-10.md)
- The statistical unit is the TREE. In both same-genus, same-day comparisons, the two trees of one species differ
  MORE than the two species do (exploratory: 2 comparisons, [03]), so no species-level claim is possible from these
  data; classify per sample from many spots, never per spectrum.
- Each species was measured on its own day, so "which day" is a shortcut for "which species". Any species result
  needs a leave-one-day-out or within-day check. The day effect is SMALLER than the species difference but
  consistent; never write "day effect > species difference".
- The silent-region control (1850–2700, no wood bands) matches or beats the fingerprint on every 9-class and
  softwood/hardwood test; only within the 0925 Chamaecyparis pair does the fingerprint lead, by a small margin.
  The control works at both spatial scales and survives baseline correction and derivatives. Report it next to
  every number.
- The export is vendor-processed and the 1600 lignin band is a dip: the bottleneck is acquisition, not the model
  (none of 36 preprocessing × classifier combos clears its silent control by more than 0.10).
- Linear models do not concentrate weight on known wood bands; the softwood/hardwood model under-weights them and
  over-weights the vendor-baseline anchor windows (955–980, 2810–2832) [05]. Another reason the raw re-export
  (before vendor baseline) is the first ask.

## Measurement schedule (each species was measured on its own day)
| date | species | power |
|---|---|---|
| 0918 | Acacia mangium | unrecorded |
| 0919 | Machilus kusanoi | unrecorded |
| 0922 | C. obtusa (JP), Fokienia (B1, B3) | unrecorded |
| 0923 | Acacia confusa, Fokienia (B1, B3 again, spots 11–15) | unrecorded |
| 0925 | C. formosensis, C. obtusa var. formosana | 24 mW |
| 0926 | S. mahagoni, S. macrophylla | 24 mW |
- Species–day aliasing: of the 36 species pairs, 4 share a measurement day, 1 is linked only through Fokienia
  (0922 + 0923), 31 are fully aliased with the day [00, `species_pairs_vs_day`]. Without assumptions these data
  cannot test (e.g. that a day shifts every species identically), those 31 cannot be separated by any algorithm —
  LLMs, pretrained spectral encoders, synthetic spectra, literature band features (already failed, [02]),
  SHAP/LIME (they show where a model looks, not why that region differs). Do NOT re-propose an algorithmic rescue
  without data in which species and day vary independently. Reasoning: docs/research-status-2026-10.md.

## Evaluation contract (non-negotiable)
1. Never report spot-level random K-fold. Every result states its split explicitly.
2. Primary: leave-one-tree-out wherever a species has >1 tree. Secondary: leave-one-block-out
   (LOBO), labelled "within-tree, optimistic". Grouping uses `block_id`/`tree_id`, never `spot`.
3. All fitted preprocessing (scaler, PCA, EMSC reference, feature selection, augmentation
   statistics) is fitted inside the training fold — use sklearn Pipelines (`woodraman.preprocess`).
4. Report balanced accuracy, macro-F1, per-class recall and the confusion matrix. Also report
   block-level accuracy (majority vote over a block's spectra) as the operational metric.
   `woodraman.evaluate.classification_metrics(y, pred, groups=block_id)` returns all of these.
5. Negative controls for every new pipeline, reported next to the main number:
   (a) within-day pair accuracy, (b) O-H-region-only, (c) **silent-region-only (1850–2700)**,
   (d) date prediction on Fokienia. If (b) or (c) is close to the main result, the pipeline is not
   measuring wood chemistry — say so in the write-up.
6. Hyperparameters are tuned with nested group CV. No tuning on held-out trees.
7. Compare every DL model against PCA-LDA / PLS-DA / linear SVM baselines under the same split.
8. Seeds fixed; each run writes `results/<date>_<exp>/metrics.json` + `config.json` (`woodraman.evaluate.write_run`).
   No result goes into a slide, paper, the proposal or THIS FILE unless it is reproducible from a committed script.
9. Metadata (date, power, tree, block, face) is never a model input. Date is used only as a split, a grouping,
   a negative-control target, or to estimate a nuisance direction that is projected OUT (EPO, in-fold).

## Code layout
- `src/woodraman/`: `io.py` (parsing + metadata), `preprocess.py` (Crop, SNV, SGDerivative, ASPLSBaseline,
  NuisanceProjector/EPO), `splits.py` (group splitters + `assert_no_leak`), `models.py` (PLS-DA + fixed
  baseline classifiers), `evaluate.py` (contract metrics + run provenance). To add: EMSC.
- `experiments/NN_<name>.py`: thin scripts importing `woodraman`; one question per script.
- `figures/`: figure scripts that only READ committed experiment results; they never recompute numbers.
- `notebooks/`: exploration only; move anything reusable into `src/`.
- `results/`, `models/`: generated and git-ignored.
- `docs/`: `research-status-2026-10.md` (current state, read first), `literature-2026-10.md`,
  `measurement-protocol-v1.md`, `methods-paper-outline-2026-10.md`. Full version for Prof. Tsao: https://claude.ai/artifact/U2hk4BiMvjXmNf88ibs96d
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
