# LOG — fusion_refine/prior（append only）

## 2026-11-24 [1] 開始：DiffLocks 分佈下的 PDF / ELBO 地圖

- 使用者想訓練處理邊界的 transformer，使結果貼近 DiffLocks 分佈；先問能否估計 3D 模型在 DiffLocks 分佈下的 PDF。
- 結論（說明給使用者）：可行。3D 髮絲 → strand VAE encoder → scalp texture（同 create_scalp_textures）→ diffusion model；
  (a) 精確 log p：`k_diffusion.sampling.log_likelihood`（probability-flow ODE + Hutchinson，慢、只適合評估）；
  (b) ELBO / 加權去噪誤差（可逐 texel 成地圖、可微）；(c) 只用 score（SDS，訓練 transformer 用）。
  注意：全域似然受低階統計主導，只做同髮型不同處理的相對比較；是 texture 空間密度（無 encode 的 Jacobian）；用無條件分支。
- 第一步：可行性檢查——原始髮型 vs 拼接（mask_only / FM 精修）的 ELBO 地圖，看邊界是否被標出來。
- 此機器沒有 DiffLocks checkpoint，需使用者以自己的帳號下載（`./download_checkpoints.sh`）。資料集也沒有 density.png → density 由髮根數估計。

## 2026-11-24 [2] (a) 精確 log-likelihood 與 (c) score 工具（checkpoint 下載中，先以隨機權重驗證程式）

- `difflocks_prior.py` 新增：
  - (a) `loglik_map`：probability-flow ODE（無條件），Heun、log-spaced σ（sigma_min → sigma_max，預設 40 步、約 80 次 forward + backward），
    Hutchinson 散度逐元素保留 → **逐 texel 的 log p 地圖**（prior + divergence；各 texel 加總 = log p 的估計），另回傳 total、bits/dim。
  - (c) `texture_builder`（固定髮根的 3D → texture，對髮絲位置可微；TBN、texel、補洞索引、density 為常數）、
    `sds_loss`（梯度 = (x − D(x + σε, σ)) / σ_data² 於 64 個 latent channel，可限 mask，σ log-uniform）、
    `sdedit`（RePaint：band 內從 σ_start 重新去噪，band 外每步換回 x + σ ε）、`decode`（DiffLocks decoder + tbn_space_to_world，可平移到原髮根）。
- 修正（不改 repo 程式）：igl 需要 int64 faces / float64 頂點；`interpolate_tbn` 回傳 torch tensor。
- smoke（`smoke_test.py`，隨機權重，val case_0000、每頭 3k 根）：所有路徑可跑；push-pull CUDA extension 編譯成功；
  sdedit 在 band 外變化 0；decode 髮根誤差 0；SDS 梯度有限、傳到髮絲位置。
  builder 與 hair_to_texture 在有髮絲的 texel 完全一致（3e-8），差異只在補洞（push-pull vs nearest）→ `hair_to_texture(fill="nearest")` 可與 builder 對齊。
  時間（隨機權重、3k 根）：texture 42 s（含首次編譯）、ELBO 2 σ × 2 noise 16 s、loglik 3 步 20 s（含編譯 / 首次呼叫）。

## 2026-11-24 [3] 交接到有權重的機器（使用者要求）

- `PLAN.md`：新機器設定、重建 cases、Step 1 ELBO 可行性 → Step 2 log p 交叉檢查 → Step 3 SDEdit baseline → Step 4 transformer + SDS，含判斷標準。
- `../nn/run.sh` 改為可攜：工作區依序找 $HAIR_WORK → 原機外接碟 → repo 上一層（標準配置）；Python 依序 $PYTHON → 工作區 venv → hair_exp。
- `../nn/models/{k1,dn1}`（SOTA 用到的權重，共 2 MB）與 `../nn/split/{split.json,lengths.json}` 放進 git；`nn_methods` 同時搜尋 `models/` 與 `$DATA/runs`。
