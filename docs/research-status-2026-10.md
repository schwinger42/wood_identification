# 研究現況與下一步（2026-10-21）

完整版（給曹老師，含圖）：https://claude.ai/artifact/U2hk4BiMvjXmNf88ibs96d
本檔是給 Claude Code / superpowers 讀的精簡版。數字都可由 `experiments/00–04` 重現（2026-10-22 全部重跑核對）。

## 一句話
現有 1,050 條光譜（9 樹種、11 株）在嚴格驗證下不能辨識樹種，連針葉 vs 闊葉也不行（留一整天：0.37，亂猜 0.5）。
主因依序：株數不足（同種不同株差異 0.46 > 不同樹種 0.38）、量測設計（樹種與量測日綁定）、儀器與匯出（背景起伏與訊號同級）。

## 能辨識什麼（主結果 = fingerprint 400–1800；對照 = 無化學波段 silent 1850–2700）
| 問題 | 驗證 | 結果 | 對照 | 亂猜 |
|---|---|---|---|---|
| 9 種，隨機切分 | spot-level random 5-fold | 0.91 | 0.93 | 0.11 |
| 9 種，換試塊 | leave-one-block-out | 0.73 | 0.74 | 0.11 |
| 9 種，換株 | held-out tree（4 株平均） | 0.43 | 0.60 | 0.11 |
| 紅檜 vs 臺灣扁柏 | held-out 臺灣扁柏的一株 | 第 1 株 0.86、第 2 株 0.39 | 第 1 株 0.68、第 2 株 0.46 | 0.5 |
| 針葉 vs 闊葉，換株 | leave-one-tree-out（11 株） | 0.62 | 0.69 | 0.5 |
| 針葉 vs 闊葉，換日 | leave-one-day-out（4 分類器平均） | 0.37（PCA-LDA 0.32，亂貼標籤 p = 0.79） | 0.39（PCA-LDA 0.42，p = 0.47） | 0.5 |
| 36 種方法最佳 | 同日樹種對，held-out tree | 0.71 | 0.69 | 0.5 |
| 同一木塊是哪天量的 | Fokienia LOBO | 0.93 | 0.98 | 0.5 |

亂貼標籤 p：把「針葉」標籤隨機貼到 9 種中的任 4 種（126 種貼法，PCA-LDA、SNV），分數不輸真標籤的比例；
p 大 = 真標籤跟亂貼沒差。來源 `experiments/02_generalisation_ladder.py`。

## 訊號預算（experiments/03；fingerprint / silent）
| 比較 | fingerprint | silent |
|---|---|---|
| 同種不同株（同日） | 0.46 | 0.31 |
| 單條光譜雜訊 | 0.39 | 0.42 |
| 不同樹種（同日） | 0.38 | 0.39 |
| 同株不同試塊 | 0.20 | 0.31 |
| 同一木塊換一天 | 0.17 | 0.33 |

## 已更正的說法
- 「量測日差異比樹種差異大」是錯的：fingerprint 約為一半（0.17 vs 0.38；silent 兩者接近，0.33 vs 0.39），
  但很一致、可被學習，且與樹種綁定 → 成為捷徑。
- 「水訊號是主因」沒有證據；含水率較可能經由螢光影響背景。
- 「無化學波段的捷徑只在細結構、寬背景只有 0.52」重現不出來（2026-10-22，`04`）：用 61/121/241 cm⁻¹ 三種寬度切開，
  寬背景 LOBO 0.68–0.71、細結構 0.71–0.75（亂猜 0.11），兩種尺度都帶捷徑。扣背景或取導數也去不掉（`01`）。

## 現有資料還能救什麼（2026-10-22 結論）
- **9 樹種辨識：救不回來，換任何演算法都一樣。** 三個天花板，都不是模型問題：
  1. 樹種與量測日綁死：36 組樹種對中，只有 4 組同一天量、1 組只能經福建柏（9/22、9/23 都量過）間接比，
     31 組的差異和量測日完全綁死（`00` 的 `species_pairs_vs_day`）。像 A 配方全在週一 1 號機台、B 配方全在週二
     2 號機台：再好的分析也分不出差異來自配方還是機台。
  2. 同種不同株差異 0.46 > 不同樹種 0.38（`03`）：要更多棵樹，模型變不出來。
  3. 驗證天花板：換一棵樹只能測 4 棵；4 棵全對，判對率的 95% 下界也只有 0.40。
- **已評估、救不了的演算法路線（沒有新資料前不要重提）：**
  - LLM 直接分類光譜：不增加資訊，同樣綁死。兩次網路搜尋沒找到 LLM 做 Raman 分類的研究（非系統性回顧）。
  - 預訓練光譜模型（RSPTE、RamanPFN 等）：帶入別處的先驗，但打不開綁死、變不出樹；檢索到的都沒有針對
    不同儀器／日期的不變性設計。若要做，只當方法論文裡的 DL 對照組，用同樣切分與對照。
  - 生成假光譜擴充：不含這些樹的新資訊，造不出株間變異，還有自我驗證的風險。
  - LLM 讀文獻抽化學峰當特徵：`02` 已證，13 個木材峰特徵不比隨機位置好。
  - SHAP／LIME：事後解釋只說模型看哪裡，不說那裡的差異來自樹種還是量測日；對線性模型就等於
    「係數 × 偏離平均」；相鄰波數高度相關使其不穩。唯一用途：論文裡畫係數光譜，當抄捷徑的圖證。
- **同一天的 4 組樹種對**（換試塊，亂猜 0.5；fingerprint / silent / O-H，`00`）：
  紅檜 vs 臺灣扁柏 0.73 / 0.58 / 0.47；兩種桃花心木 0.83 / 0.81 / 0.92；日本扁柏 vs 福建柏 0.72 / 0.72 / 0.74；
  相思樹 vs 福建柏（唯一同日針葉 vs 闊葉）0.91 / 0.88 / 0.90。只有第一組化學峰領先。
  推論（未驗證）：同一天內 silent 區的可分性不可能來自量測日，可能是樣品螢光（木質素）或表面狀態；
  每種只有 1 棵樹，分不出是株還是種的特性。
- **日期永遠不當模型輸入**（CLAUDE.md 評估規則第 9 條）。缺的是「讓模型能學會忽略日期」的量測
  （同一樹種跨日、每日參考樣品），不是日期特徵。
- **還能做的：** 方法論文（現在）；同日樹種對當可行性分析；原始資料重新匯出（不算新量測）；
  最小補量方案（`docs/measurement-protocol-v1.md` 開頭）。

## 下一步
1. 向曹老師要：raw 重新匯出（同一批量測）、量測順序與時間、儀器參數、樣品來源與含水狀態（清單見完整版第 8 節）。
2. raw 到手後：重跑 `00`、`03`、`04`，看 1600 cm⁻¹ 峰（現在是凹谷）與背景起伏是否改善。
3. 用現有資料起草方法論文（驗證陷阱 + 負對照）。
4. 第二期：每樹種 ≥5 株（理想 8–10）、跨日平衡區集設計、每日參考木＋Si＋SRM 2242a、樣品調濕或烘乾。

## 詳細數字（2026-10-22 從 CLAUDE.md 原文搬來；方括號 = 產生該數字的實驗腳本）
- Exported data looks vendor-processed: ~3% negative values; 78% / 81% of spectra are <= 0 inside
  955–980 / 2810–2832 cm-1 → likely fixed-anchor baseline subtraction by the instrument software (INFERENCE).
  Raw export is NOT in the repo; ask Prof. Tsao. [00]
- **The dominant structure is not wood chemistry.** One shared shape explains a median 74% of each
  spectrum's variance; between-species median correlation is 0.84 overall and 0.83 in the Raman-silent region
  1850–2700 cm-1 (no wood bands there) [00]. Each day's median spectrum correlates 0.90–0.97 with the global
  median, and the best axis shift per day is ≤ 0.15 cm-1 [04] → instrument-fixed pattern (INFERENCE).
- Real wood bands ARE present but small (global median, 400–3450): peaks at 1327, 1457, 1661, 1107 and 2957
  (prominence 777/538/727/538/683) lie within 20 cm-1 of an expected band, yet bumps of the same size sit in the
  silent region (2124/2259/2490/2628: 536–625) and the biggest peaks (1016, 648, 832) match no wood band.
  1600 (lignin, normally the strongest band) is a DIP (local minimum 1597.8). An unexplained peak at 2786
  (prominence 754) sits just below the 2810–2832 vendor anchor — INFERENCE: possibly a baseline artefact,
  which would also distort the low edge of the C-H band. [04]
- The pattern's overall amplitude alone does NOT predict species (LOBO 0.16, chance 0.11); the shortcut is in
  shape, not scale. Offsets of the 5 near-band peaks from band centres are mixed in sign (−3.5, −2.7, +0.8,
  +12.1, +17.3 cm-1): no consistent axis offset. [04]
- Session effect: on the same two Fokienia blocks, date is predictable on the held-out block
  (balanced acc 0.93 fingerprint, 0.98 silent region; only 2 blocks, so weak evidence but consistent). [00]
- **Signal budget (experiments/03, RMS difference of group-mean SNV spectra, debiased; fingerprint / silent):**
  tree vs tree same species same day 0.46 / 0.31 · species vs species same day 0.38 / 0.39 ·
  single-spectrum noise 0.39 / 0.42 · block vs block same tree 0.20 / 0.31 · same wood, other day 0.17 / 0.33.
  → Between-tree variation EXCEEDS the species difference (main reason held-out trees fail; needs more trees).
  → The day effect is SMALLER than the species difference in magnitude (~half in the fingerprint, 0.17 vs 0.38;
    in the silent region they are close, 0.33 vs 0.39) but consistent, hence learnable;
    with species nested in days it becomes a shortcut. Do NOT write "day effect > species difference".
  → Single spectra are as noisy as the species difference: classify per sample from many spots, not per spectrum.
  Tiny n behind each row (tree 2 pairs, species 3, day 2): read as orders of magnitude.
- Silent region 1850–2700 predicts species as well as the fingerprint (LOBO 0.74 vs 0.73, chance 0.11) and
  better on held-out Swietenia trees (0.84/0.84 vs 0.62/0.42). O-H region alone: LOBO 0.69. [00]
- Held-out-tree recall (9-class, fingerprint): C. obtusa var. formosana 0.33/0.33, S. macrophylla 0.62/0.42. [00]
- Inflation ladder (9-class PCA-LDA, fingerprint / silent): spot-level random 5-fold 0.91 / 0.93 →
  LOBO 0.73 / 0.74 → held-out tree 0.43 / 0.60. Quote spot-level only as this demonstration. [02]
- The silent-region shortcut sits at BOTH spatial scales: splitting the SNV silent region with a 61/121/241 cm-1
  boxcar, broad structure gives LOBO 0.68–0.71 and narrow 0.71–0.75 (chance 0.11); Fokienia date 0.78–0.97
  from either [04]. (An earlier unscripted claim, "narrow only, broad LOBO 0.52", did not reproduce.)
  Baseline correction and derivatives leave the control high: with asPLS / SG1 / SG2, silent within-day
  held-out-tree recall 0.60–0.74 and silent Fokienia date 0.68–1.00 (both chance 0.5) [01].
- Method sweep (6 preprocessings × 6 classifiers incl. RF/MLP, untuned): best within-day
  held-out-tree recall 0.71 (SG1+SNV+RF) vs its silent control 0.69; max fingerprint−silent margin 0.10.
  C. obtusa var. formosana tree 2 is never recognised (0.12–0.47 in all 36) — it looks like C. formosensis.
  On the 0925 pair alone, fingerprint beats silent in 29/36 methods, mean margin 0.08.
  EPO from Fokienia session pairs removes the date signal but does not improve species generalisation.
  Picking the best of 36 on 4 test trees is itself selection bias. With 4 held-out trees, even 4/4 has a
  95% CI of [0.40, 1.00]: no method can make a species-level claim from this data. [01]
- **Unseen-day test:** softwood vs hardwood (chemically the easiest split), leave-one-DAY-out:
  balanced acc fingerprint 0.37, silent 0.39, C-H 0.23, O-H 0.18 — BELOW chance 0.5 — vs leave-one-tree-out
  fingerprint 0.62, silent 0.69 (the control is HIGHER). Exact permutation test (PCA-LDA, all 126 ways to call
  4 of 9 species "softwood"): p = 0.79 fingerprint, 0.47 silent, 0.80 C-H, 0.99 O-H — the real labels do no
  better than arbitrary ones. Whatever the models learn does not survive a new measurement day. [02]
- Known-band features (13 wood bands) do no better than the same features at random off-band fingerprint
  positions. [02]
- Within-day pairs are the only species comparisons not confounded by session. Best evidence of chemistry:
  0925 Chamaecyparis pair, LOBO fingerprint 0.73 > silent 0.58 > O-H 0.47 (chance 0.5) — but held-out tree
  0.86 / 0.39. In the 0926 Swietenia pair, O-H (0.92) and silent (0.81) match or beat fingerprint (0.83).
  0922 C. obtusa vs Fokienia: 0.72 / 0.72 / 0.74. 0923 Acacia confusa vs Fokienia (the ONLY same-day softwood vs
  hardwood pair): fingerprint 0.91, silent 0.88, O-H 0.90 — separable, but not via bands. [00]
