# CLAUDE.md — inference/fusion_refine/prior/

DiffLocks 先驗（texture 轉換、ELBO / score）用於融合邊界。概念見 README；進度只寫在 LOG（append only）、SUMMARY（「目前狀態」≤ 20 行）、checkpoints（Promising / Experimental / Baseline / 已棄用）。

- 執行：`bash ../nn/run.sh ../prior/<script>.py`（同 nn：hair_work 工作區、$DATA、PYTHONPATH）。需要 `checkpoints/strand_vae`、`checkpoints/difflocks_diffusion`（`./download_checkpoints.sh`）。
- 與訓練資料一致的轉換：strand 世界座標 → 髮根 TBN（scalp.ply）→ normalize → `StrandCodec.encoder` → 64 ch latent 放到 (row, col) texel → 補洞 → density channel
  （(d − 0.5) / (0.5 / sigma_data)）。此資料集沒有 density.png，density 由髮根數估計（與原始定義可能不同，只做相對比較）。
- 模型以無條件分支評估（不給 latents_dict = condition dropout 的 dummy 條件）。
- 不改 DiffLocks / benchmark 程式；評估方式與結論都可能改變，不要寫死。
