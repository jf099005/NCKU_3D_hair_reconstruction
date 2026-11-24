# fusion_refine/nn

用神經網路（flow matching / diffusion 等生成模型）取代 `fusion_refine/` 裡的兩個手工步驟：
**選髮絲**（joint FM：每個 texel / 髮絲留哪一頭）與**小幅位移**（側向梳理 `comb.py`）。

概念：兩顆頭（`{"positions", "root_uv"}`）與 drawn mask（scalp-texture layout）先轉成 scalp texture 上的條件特徵圖，
模型在同一個 texture 空間輸出 refined mask 與位移場，再套回髮絲（保長、髮根固定）。
特徵、輸出、模型與監督來源（teacher 蒸餾 / 自監督 / reward）都在實驗中，會改；目前用哪個版本見 `checkpoints.md`。

- 進度與結果：`LOG.md`（逐步紀錄）、`SUMMARY.md`（目前狀態）、`checkpoints.md`（可用版本）
- 執行方式與慣例：`CLAUDE.md`
