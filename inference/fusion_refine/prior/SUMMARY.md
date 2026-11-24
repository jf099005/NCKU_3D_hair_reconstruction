# SUMMARY — fusion_refine/prior

## 目前狀態

- 2026-11-24 開始：以 DiffLocks 先驗衡量 / 改善拼接邊界。
- 工具（`difflocks_prior.py`，LOG [1]、[2]）：3D → scalp texture、ELBO 地圖、逐 texel log-likelihood（a）、可微 texture + SDS 與 SDEdit 邊界重去噪（c）、decode。
  已用隨機權重驗證程式路徑；尚未用真實權重跑任何數字。
- 改到另一台已有權重的機器繼續，步驟見 `PLAN.md`（LOG [3]）。
- 下一步：`elbo_check.py`（原始 vs 拼接的 ELBO / log p 地圖，看邊界是否被標出）→ SDEdit 邊界 baseline → transformer + SDS。

## 重要結果

（尚無）
