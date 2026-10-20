# Literature map — Raman + ML wood species ID (re-checked 2026-10-21)

Status: ✅ checked on the publisher/arXiv page · ⚠️ partly checked (see note).
Coverage caveat: web search only, no Scopus/WoS. "Nothing found" means nothing found by web search,
not proof of absence — confirm the gap with a Scopus/WoS query before writing it into a paper.

## 1. Raman for wood species (direct prior art — thin)
- ✅ Sowoidnich, Towrie, Matousek 2023, *Appl. Spectrosc.* 77(6):666–681, doi:10.1177/00037028231168405 —
  830 nm SERDS, 1 mm wide-area spot, sample rotation; balsa/beech/birch/hickory/pine; PLS-DA 99.4%.
  Validation split unit not stated in the abstract (full text not reachable) — check before citing the number.
- ⚠️ "Raman spectroscopy for identification of wood species", *J. Phys. Conf. Ser.* 741:012131 (2016),
  doi:10.1088/1742-6596/741/1/012131 — qualitative; author list not confirmed.
- ✅ Jain, Rose, Jayapal, Singh, Ram 2024, *Sci. Rep.* 14:12692, doi:10.1038/s41598-024-62932-0 —
  in-vivo leaf clip on living plants, 830 nm, 11 species × 3 plants; the 91% LDA figure is within plant
  groups (fern/monocot/eudicot), not 11-way species ID; 13 test plants. Water not discussed.
- ⚠️ Farber, Sanchez, Kurouski 2020, *RSC Adv.* (doi:10.1039/d0ra03697h) — handheld 830 nm, PLS-DA, leaves of >15 species.
- No 2024–2026 Raman + ML wood-species paper and no public Raman wood spectral database found.

## 2. Evaluation rigour (cite to defend tree-level splits)
- ✅ Rosa da Silva N., Deklerck V., Baetens J.M. et al. 2022, *Plant Methods* 18:79,
  doi:10.1186/s13007-022-00910-1 — wood images: 4-fold image-level CV 0.95 vs leave-k-trees-out 0.22–0.30;
  over-prediction "up to 60%".
- ✅ Casanova, Gontijo, Candido Jr., Nisgoski, de Paula Filho 2026, arXiv 2610.12676 (WoodBR-Cross; preprint) —
  303 trees; on the 20-class subset (214 trees, ConvNeXt-Tiny) 0.974 image split vs 0.862 tree-disjoint
  (authors note the two scores are not strictly the same metric).
- ✅ Ravindran et al. 2021, *Front. Plant Sci.* 12:647515, doi:10.3389/fpls.2021.647515 — 97% 5-fold CV vs
  86.5% top-1 on independent xylarium specimens.
- ✅ Guo, Bocklitz, Neugebauer, Popp 2017, "Common mistakes in cross-validating classification models",
  *Anal. Methods* 9:4410–4417, doi:10.1039/C7AY01363A — split at the highest hierarchy level; refit in-fold.
- ✅ Beleites, Neugebauer, Bocklitz, Krafft, Popp 2013, *Anal. Chim. Acta* 760:25–33,
  doi:10.1016/j.aca.2012.11.007 — ~75–100 independent test cases to test one classifier; hundreds to compare two.
- ✅ da Silva W.D.S. et al. 2025, *Forests* 16:984, doi:10.3390/f16060984 — NIR, 4 species × 4 trees, CV blocked
  by increment core (not strictly by tree); PLS-DA 98.5% ≫ kNN 57%.

## 3. Moisture, fluorescence and the 532 nm problem
- ✅ Kunze et al. 2021, *Microchem. J.* 171:106839, doi:10.1016/j.microc.2021.106839 — NIR; oven-dry model on
  ambient samples: 87.8% inconclusive → 18.4% with EPO; extreme moisture 100% → 43.8% with EPO.
- ✅ Gomes, Medeiros, Viana, Hein 2025, *Holzforschung* 79(4–5):188–201, doi:10.1515/hf-2024-0111 — NIR,
  11 species × 10 specimens, 5 moisture stages; per-stage 86–100%, global model ~85% benchtop / ~75% portable.
- ✅ Liland, Kohler, Afseth 2016, *J. Raman Spectrosc.* 47(6):643–650, doi:10.1002/jrs.4886 — EMSC for Raman.
- ✅ Peters & Rapp 2024, *Cellulose* 31:4149–4160, doi:10.1007/s10570-024-05898-6 — moisture quenches and
  blue-shifts lignocellulose fluorescence; strongest quenching at Ex 545 / Em 635 nm. With 532 nm excitation,
  635 nm ≈ 3070 cm⁻¹ shift, so the moisture-dependent background sits under the whole Raman window.
- ✅ Lähdetie et al. 2013, *Holzforschung* 67(5):531–538, doi:10.1515/hf-2012-0177 — laser-induced fluorescence
  of lignin at 532 nm.
- ✅ Agarwal 2019, *Molecules* 24:1659, doi:10.3390/molecules24091659 — review: visible excitation gives
  significant lignin fluorescence; 785/1064 nm largely avoid it; D₂O/H₂O exchange method.
- ✅ Prats-Mateu et al. 2018, *Sci. Rep.*, doi:10.1038/s41598-018-30096-3 — 532 nm laser alters spruce lignin
  bands (1600/1660) and fluorescence, in water and in air → laser dose must be fixed and recorded.
- ⚠️ Masson et al. 2018, *Analyst*, doi:10.1039/c8an01989d — 680 + 785 nm used to MEASURE hydration from the
  O-H band (3035–3680 cm⁻¹); a different "two-laser" idea from SERDS.
- ⚠️ Ponzecchi, Thybring, Fredriksson et al. — Raman of wood in a humidity chamber with D₂O (venue/year unconfirmed).
- Water is a weak Raman scatterer: O-H band cross-section 8.2×10⁻³⁰ cm²/sr vs 33–75 for benzene/cyclohexane C-H
  on the same scale (Burns et al. 2020, *Anal. Chem.* 92:10686).
- SERDS (two slightly shifted lasers) cancels backgrounds that do not shift with excitation (fluorescence,
  fixed pattern, etalon). Water's O-H and H-O-H bands are Raman bands, shift with the laser, and survive SERDS
  (inference from the principle; no source discusses water under SERDS directly).
- Gierlinger, Keplinger, Harrington 2012, *Nat. Protoc.* 7(9), doi:10.1038/nprot.2012.092 — confocal Raman
  imaging protocol for plant cell walls (sample prep reference).
- Gap: no Raman-wood species paper models moisture explicitly.

## 4. Preprocessing
- pybaselines (asLS, airPLS, arPLS, asPLS, …); ✅ RamanSPy, Georgiev et al. 2024, *Anal. Chem.* 96:8492–8500,
  doi:10.1021/acs.analchem.4c00383 — different pipelines give substantially different results →
  report preprocessing sensitivity.

## 5. DL at small n
- ✅ Ho et al. 2019, *Nat. Commun.* 10:4927, doi:10.1038/s41467-019-12898-9 (Bacteria-ID) — 1D ResNet,
  pretrain/fine-tune, test on separately cultured isolates.
- ✅ Liu, Wu, Wang, Qi, Zhou, Xue 2026, *Sensors* 26:341, doi:10.3390/s26010341 (review) — on a 12k-spectrum set,
  1D-CNN won on full data; PLS-DA won when spectra were subsampled.
- ✅ Soysal et al. 2025, *ACS Earth Space Chem.*, doi:10.1021/acsearthspacechem.5c00136 (arXiv 2509.25964) —
  compact 1D CNN; contrastive/semi-supervised pretraining +≤11% at 10% labels.
- ✅ RamanPFN, arXiv 2608.02157 (preprint, v2 Sep 2026) — 74 datasets; no small-n gain figures in abstract.
- ✅ RSPTE, Eranti et al., *IEEE Access* 2026, doi:10.1109/ACCESS.2026.3672109 — Barlow Twins pretraining;
  20% data: 50–86% with vs 9–57% without pretraining.
- ⚠️ Garzón et al. 2026, *Eng. Appl. AI* 167:113593 — few-shot NIR/MIR benchmark (not Raman).
- Verdict: at ~9 classes × 1–2 trees, expect no reliable DL gain over PCA-LDA/PLS-DA/SVM. More importantly,
  our own audit (below) shows the bottleneck is acquisition, which no model choice fixes.

## 6. Taiwan context
- Hung et al. 2012 (NCHU) FTIR-ATR of 40 Taiwanese timbers (no validation) — ⚠️ not re-checked.
- Investigation Bureau uses SSR DNA for Taiwan red cypress vs Taiwan hinoki in illegal-logging cases —
  ⚠️ not re-checked; a rapid Raman screen would address a real forensic gap.
- Swietenia macrophylla / mahagoni / humilis: CITES Appendix II.

## 7. Our own audit (experiments/00_confound_audit.py, 2026-10-21)
- The Raman-silent region 1850–2700 cm⁻¹ (no wood bands) predicts species as well as the fingerprint region
  (LOBO 0.74 vs 0.73) and better on held-out Swietenia trees (0.84 vs 0.42–0.62). One shared shape explains
  a median 74% of each spectrum's variance. → The current data mostly carry instrument/background/session
  structure, not wood chemistry. Fix acquisition before modelling.

## Candidate contributions (unchanged, now with a prerequisite)
0. Prerequisite: clean acquisition (raw export, dark, intensity calibration, fixed laser dose, dried samples)
   and ≥3 (target 5) trees per species measured in interleaved sessions.
1. Hierarchical split audit + variance decomposition (species > tree > block > face > spot) with
   silent-region and session negative controls.
2. Moisture-robust Raman via a paired drying series: dry-only vs fingerprint crop vs EMSC(water) vs EPO vs augmentation.
3. Face (C/R/T) as domain shift.
4. Controlled DL-vs-chemometrics comparison with learning curves.
5. Open, metadata-rich Raman wood dataset (Zenodo).
