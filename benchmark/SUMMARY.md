# SUMMARY — fusion mask 精修

逐步紀錄見 `LOG.md`，可用版本見 `checkpoints.md`。

## 目前狀態

- 目標：讓 `img2hair_tsai_*` 的融合適配更多髮型 / mask；主要問題是重疊大（mid/long）時 FM 吃掉一顆頭或改變髮型。
- r5（全髮絲 4 case + highlighting optical 渲染）完成：baseline 3/4 case 嚴重違反 mask；`afm_a2_guard0.7` 4/4 維持 mask 但重疊大時幾乎不精修；
  lift 類在 optical 渲染下也無效（已棄用）。
- 評估：gs + ret + view_mix + optical 渲染（`render_results.py --optical`，需全髮絲 run）。
- 下一步：方向 A — 依髮流側向梳回入侵髮絲 / 對齊融合邊界；以及 guard 與 baseline 之間的中間設定。

## 重要結果

- optical 渲染（r5）：baseline 的低分多來自「改髮型」（一頭消失 / donor 擴張）；guard FM 是唯一 4/4 守住 mask 的版本（LOG [8]）。
- strand editing（徑向 lift）無效：雜亂來自髮根附近橫躺在對方區域表層的入侵髮絲（LOG [7]）。
- FM 的本質上限：重疊大的 case 只靠保留 / 藏起，不藏掉大半顆頭就消不掉交叉（LOG [4]）。
- r1（20k，8 case）：gs 與 ret 強烈取捨；baseline 5/8 case 某頭 < 50%；guard 版 0/8。見 LOG [3]。
- Baseline（全髮絲 50 case）：gs 0.170 → 0.048，但 ret 平均 0.70，8/44 case 某頭 ret ≤ 0.1。
