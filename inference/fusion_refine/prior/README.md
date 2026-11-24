# fusion_refine/prior

以 DiffLocks 本身（strand VAE + scalp-texture diffusion）當作「頭髮分佈」的先驗，用來評估與平滑兩顆頭拼接的邊界。
概念：融合後的 3D 髮絲照 DiffLocks 建資料的方式轉成 scalp texture（每根 latent 放到髮根 texel + density），
再用 diffusion model 的去噪誤差（ELBO）/ score 衡量「離 DiffLocks 分佈多遠」，之後可作為邊界處理（SDEdit / transformer + SDS 等）的目標。

- 進度：`LOG.md`、`SUMMARY.md`、`checkpoints.md`；執行方式：`CLAUDE.md`
