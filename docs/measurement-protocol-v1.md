# 下一批量測：protocol v1（草案，待曹老師確認）

> 2026-10-21 更新：PoC 階段無法追加量測，現有 1050 條光譜就是全部。
> - **現在就能做、而且不算新量測**：A 段的問題，以及請儀器軟體把「同一批量測」重新匯出成 raw
>   （沒做 baseline）、附上量測順序和時間戳記。
> - B–I 段改當作經費申請書裡的資料計畫，也就是拿到錢之後要怎麼量。

為什麼要先改量測：`experiments/00_confound_audit.py` 顯示，在沒有木材 Raman 峰的 silent region
（1850–2700 cm⁻¹）也能「辨識樹種」，準確率跟 fingerprint 區差不多（LOBO 0.74 vs 0.73）。所以目前分類器學到的
多半是儀器、背景、量測日的狀態，不是木材化學。這一點換模型解決不了，要從量測端修。

## A. 先問曹老師（會影響後面的設計）
1. 儀器型號、光柵、物鏡、積分時間、累加次數；樣品上的實際雷射功率（沒有 `24mW` 前綴的檔案是多少？）。
2. 能不能匯出 **raw**（沒做 baseline、沒 smoothing）的光譜？同一次量測能不能附 dark spectrum？
3. 軟體匯出時預設做了什麼處理？（資料在 955–980 和 2810–2832 cm⁻¹ 都被壓到 ≤0，看起來是固定錨點的 baseline。）
4. 每天量測時樣品的含水狀態（新鮮/濕、氣乾、烘乾）？0922 和 0923 之間 Fokienia 樣品放在哪裡？
5. 「水訊號蓋過去」在螢幕上看到的是什麼？是 3000–3600 cm⁻¹ 的寬峰，還是整條光譜被抬高的螢光背景？
6. `Chamaecyparis obyusa` 資料夾是不是日本扁柏（C. obtusa）？
7. 有沒有 785 nm 雷射可以借用，或校內/合作單位有沒有 1064 nm FT-Raman？

## B. 每個 session 的儀器檢查
- 開始和結束各量一次 Si wafer（520.7 cm⁻¹），記錄峰位和強度，用來校正 wavenumber 和監控漂移。
- 量 dark spectrum（雷射關閉、條件相同）。
- 如果有強度標準品（白光燈或 532 nm 用的 luminescence standard，例如 NIST SRM 2242a），每天量一次，
  用來校正儀器響應。目前光譜裡有一個所有樣品共用的週期性起伏，這一步可能就能把它除掉。
- 所有參數固定並寫進 metadata：樣品上功率、積分時間、累加次數、物鏡、對焦方式。

## C. 樣品準備
- 含水率先固定，這是處理「水的問題」最便宜的方法：全部樣品在 20 °C / 65% RH 下調濕到平衡（約 12% MC），
  或烘乾（103 ± 2 °C）後放乾燥箱。量測前後各秤一次重量並記錄。
- 用刀片或切片機切出新鮮平整的表面，同一批樣品用同一種處理方式。
- 主要量測面先固定一個（建議 R 或 T），C/R/T 三面全量的設計留給「face = domain shift」那個子題。

## D. 螢光與雷射傷害（532 nm 的已知問題）
- Pilot 先做：同一個點連續量 10–20 次，看背景什麼時候下降到穩定（photobleaching），也看 1600/1660 cm⁻¹
  峰比會不會變（Prats-Mateu 2018：532 nm 會改變木質素的峰）。依結果定出固定的 pre-bleach 時間和功率上限，
  之後所有樣品都照這個做。

## E. 取樣設計：要多棵樹，不要同一棵樹的很多點
- 獨立樣本是「樹」。同一棵樹切 3–4 塊、每面量 10 點，對 species-level 的結論幾乎沒有加分。
- 目標：每個樹種 ≥5 棵樹（最少 3 棵），每棵樹 1–2 塊，每塊 10 點。
- 樣本來源：木材標本館（xylarium，例如林試所、各校森林系的標本）、有產地證明的木材行。
  只需要幾 mm 大小的平整面，可以跟標本館談小塊取樣或非破壞量測。
- Phase 1 範圍要縮小：先做 **紅檜（C. formosensis）vs 臺灣扁柏（C. obtusa var. formosana）**，各 5 棵樹。
  這組有盜伐鑑識的應用價值，而且是同屬，難度足夠。Swietenia（CITES II）當作第二組。

## F. Session 設計：每天的量測要涵蓋所有樹種
- 每個量測日都要量到所有樹種，順序隨機，至少分散在 3 天。這樣「日期」和「樹種」才不會綁在一起。
- 記錄日期、時間、操作者、室溫、RH。

## G. 含水率配對實驗（可以發展成方法學貢獻）
- 每個樹種挑 1–2 塊，量 wet → 氣乾 → 調濕平衡 → 烘乾 四個狀態，同一點位重複量。
  拿來比較 fingerprint crop、EMSC（把水光譜當 interferent）、EPO、augmentation 對含水率的穩健性。

## H. Metadata（每條光譜一列，CSV）
`file, species, tree_id, source(xylarium/vendor/field), block, face, spot, date, time, session_order,
operator, laser_nm, power_mW_at_sample, integration_s, accumulations, objective, prebleach_s,
moisture_state, mass_g, room_T, room_RH, raw_export(bool), notes`

## I. 統計上要誠實
- 10 棵樹做 leave-one-tree-out，準確率的信賴區間會很寬（Beleites 2013：要比較兩個分類器，大約需要數百個
  獨立測試樣本）。Phase 1 要主張的是「方法可行，加上一條 learning curve 說明需要多少棵樹」，
  不是一個 99% 的數字。這正好可以寫成經費申請的理由。
