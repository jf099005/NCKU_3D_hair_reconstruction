# SUMMARY — fusion mask 精修

逐步紀錄見 `LOG.md`，可用版本見 `checkpoints.md`。

## 目前狀態

- 目標：讓 `img2hair_tsai_*` 的融合適配更多髮型 / mask；重疊大（mid/long）時 FM 會吃掉一顆頭或改變髮型。
- Promising：`comb20_slab1`（深度規則 FM + 側向梳理 2 cm；28 case view_mix 達 baseline 水準且 0 崩）、`afm_a2_guard0.7`（及結果相同的 `_fast`，全髮絲約 100 s）。
- 難度主要由「入侵比例」決定（LOG [19]）；深度分離的交叉在長髮佔約 60%（LOG [20]）。
- 下一步：近距離渲染檢查梳理是否自然；入侵比例做為何時用積極方法的準則；回寫 img2hair_tsai_1.py（預設關閉的 flag，需使用者同意）。

## 重要結果

- optical 渲染（r5）：baseline 的低分多來自「改髮型」（一頭消失 / donor 擴張）；guard FM 是唯一 4/4 守住 mask 的版本（LOG [8]）。
- strand editing（徑向 lift）無效：雜亂來自髮根附近橫躺在對方區域表層的入侵髮絲（LOG [7]）。
- 深度規則 + 側向梳理（`comb20_slab1`）：第一個在 view_mix 追上 baseline 且 0 崩的版本（LOG [20]、[21]）。
- FM 的本質上限：重疊大的 case 只靠保留 / 藏起，不藏掉大半顆頭就消不掉交叉（LOG [4]）。
- r1（20k，8 case）：gs 與 ret 強烈取捨；baseline 5/8 case 某頭 < 50%；guard 版 0/8。見 LOG [3]。
- Baseline（全髮絲 50 case）：gs 0.170 → 0.048，但 ret 平均 0.70，8/44 case 某頭 ret ≤ 0.1。
