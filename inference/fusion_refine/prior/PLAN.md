# PLAN — 換到有 DiffLocks 權重的機器後的實驗計畫（2026-11-24）

目標：讓兩顆頭拼接的邊界平滑、且貼近 DiffLocks 的頭髮分佈。路線：先用 DiffLocks 先驗**量測**邊界（ELBO / log p 地圖），
確認量測有效後，做不需訓練的 **SDEdit 邊界重去噪**，最後訓練**邊界 transformer + SDS loss**。
進度紀錄照舊寫在本資料夾的 `LOG.md`（append）、`SUMMARY.md`、`checkpoints.md`。

---

## 0. 目前狀態（交接時）

**選髮絲 + 位移（`../nn/`，見 `../nn/SUMMARY.md` 的 SOTA 表）**

| 標記 | 方法 | 說明 | test（12 case，mid / long，10k 髮絲）gs / view_mix / slab_w | 秒 / case |
|---|---|---|---|---|
| SOTA 品質 | `dopt_fm_mix10` | FM 選髮絲 + 可微混雜目標直接最佳化 | 0.091 / 0.0437 / 4.2 | 41 |
| SOTA 全 NN | `dnet_dn1_k1` | NN 選髮絲（k1）+ 無標註攤銷位移 NN（dn1） | 0.110 / 0.0447 / 33.8 | 1.9 |
| 參考 | `comb20_slab1` | FM + 手工梳理 | 0.091 / 0.0450 / 14.1 | 39 |

k1 / dn1 的權重在 `../nn/models/{k1,dn1}/`（git 內，`nn_methods` 會自動找到）；資料切分在 `../nn/split/`。

**DiffLocks 先驗工具（本資料夾 `difflocks_prior.py`）——只用隨機權重驗證過程式路徑，還沒有任何真實數字**

| 功能 | 函式 | 說明 |
|---|---|---|
| 3D → texture | `encode`、`hair_to_texture(fill=...)` | 與 `data_processing/create_scalp_textures.py` 相同：髮根 TBN → normalize → `StrandCodec.encoder` → (row, col) texel → push-pull 補洞 → density |
| ELBO 地圖 | `elbo_maps` | 各 σ 的逐 texel 去噪誤差（無條件分支），固定亂數種子，不同 texture 可直接比 |
| (a) 精確 log p | `loglik_map` | probability-flow ODE（Heun、40 步 ≈ 80 次 fwd + bwd），Hutchinson 散度逐元素保留 → 逐 texel log p 地圖 + total + bits/dim |
| (c) 可微 texture | `texture_builder` | 固定髮根時 3D → texture 對髮絲位置可微（補洞用 nearest） |
| (c) SDS | `sds_loss` | 梯度 = (x − D(x + σε, σ)) / σ_data²，可限 mask（邊界帶） |
| (c) SDEdit | `sdedit` | RePaint：band 內從 σ_start 重新去噪，band 外每步換回 |
| decode | `decode` | latent → 世界座標髮絲（可平移到原髮根） |

已知差異（只影響絕對數值，相對比較不受影響）：此資料集副本沒有 `density.png`，density 由髮根數估計；
texture 需要接近全部髮絲（10k 根時 92% texel 是補洞 → 本身就 OOD），所以**先驗相關實驗一律用全髮絲**（`max_strands` 不設）。

---

## 1. 新機器設定

1. 取得程式：`git fetch && git checkout adaptive_fuse_v0`（本計畫所在 branch）。
2. 目錄配置（標準）：`<P76154862>/{NCKU_3D_hair_reconstruction, highlighting, DiffLocks_Dataset/difflocks_dataset}`。
   `../nn/run.sh` 會依序找 `$HAIR_WORK` → 原機外接碟 → **repo 的上一層**，所以標準配置不需設定；否則 `export HAIR_WORK=<放 highlighting 與 DiffLocks_Dataset 的資料夾>`。
3. 權重：`checkpoints/strand_vae/strand_codec.pt`、`checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth`（repo 根目錄）。
4. Python：`$PYTHON` → `<工作區>/venv` → conda env `hair_exp`。需要 torch、NATTEN、flash-attn、trimesh、igl、torchdiffeq、numba、scipy、cv2；push-pull extension 首次執行時以 nvcc 編譯。
5. `$DATA`（預設 `<工作區>/fusion_nn_data`，不進 git）：
   ```bash
   mkdir -p $DATA && cp inference/fusion_refine/nn/split/{split.json,lengths.json} $DATA/
   ```
6. 程式路徑檢查（不需權重）：`bash inference/fusion_refine/nn/run.sh ../prior/smoke_test.py`（需先有 `$DATA/val`，見第 2 步）。

## 2. 重建 benchmark cases（case.json 存的是絕對路徑，不能直接搬）

用同一個 split 與 seed（`--names_from` 只取 split 中的髮型；新機器資料集較完整時也只用這 189 個裡的對應部分）：

```bash
cd <repo>
R=inference/fusion_refine/nn/run.sh
bash $R build_cases.py 16 --seed 21 --min_hair_length 0.03 --names_from $DATA/split.json:val  --out $DATA/val
bash $R build_cases.py 12 --seed 31 --min_hair_length 0.03 --names_from $DATA/split.json:test --out $DATA/test
```

若資料集內容與原機相同，case 會完全相同（可比對 `case.json` 的髮型名稱與 LOG 的數字）。接著跑參考方法（10k 髮絲，約 40 s / case）：

```bash
bash $R bench.py $DATA/val --run_name teacher10k --max_strands 10000 --no_render --no_figures \
  --methods mask_only,afm_a2_guard0.7_slab1,comb20_slab1,dopt_fm_mix10,dnet_dn1_k1 --metrics grid_score,strands,view_mix,slabx
```

（`--max_mem_gb` 依機器設；全髮絲 FM 需 ~12 GB RAM。）

## 3. 實驗步驟

### Step 1 — ELBO 地圖可行性（最優先，約 3–4 case × 數分鐘）

```bash
bash $R ../prior/elbo_check.py $DATA/val --cases case_0000,case_0002,case_0007,case_0011 \
  --refined_run $DATA/val/runs/teacher10k --refined_method comb20_slab1 --out $DATA/prior/elbo_v1
```

每個 case 比較：原始 base、原始 donor、drawn mask 拼接（mask_only）、FM 精修 mask 拼接（全髮絲）。輸出 `stats.json`（邊界帶 vs 其餘區域的平均誤差與比值）與 png。

判斷標準：
- 有效：mask_only 的邊界帶比值（band / rest）**明顯高於**原始髮型在同一條線上的比值，且精修後下降 → 量測對「拼接邊界」有反應。
- 若原始髮型在邊界帶也一樣高（例如受密度 / 補洞影響）→ 先檢查 density 估計（改 σ、改正規化）與 `fill="nearest"`，再判斷。
- 看哪個 σ 最敏感（預期 0.1–0.3）；之後的 SDS / SDEdit 以此選 σ 範圍。

### Step 2 — (a) 精確 log p 交叉檢查（1–2 case，每個 texture 數分鐘）

對 Step 1 的同一組 texture 跑 `prior.loglik_map(tex, n_steps=40)`，比較 total / bits-per-dim 與逐 texel 地圖是否和 ELBO 地圖一致。
注意：全域 log p 受低階統計主導，只比較同一 case 的不同版本；texture 間用相同 Hutchinson 種子。若與 ELBO 一致，之後只用較便宜的 ELBO。

### Step 3 — SDEdit 邊界重去噪 baseline（不需訓練）

待寫：`sdedit_fuse.py`（或 benchmark 方法）
1. 融合結果（keep 來自 mask_only / FM；位移版可之後再加）→ `hair_to_texture`（全髮絲）。
2. band = drawn（或 refined）邊界膨脹 w texel（試 4 / 8 / 16）∩ 頭髮區域。
3. `prior.sdedit(tex, band, sigma_start ∈ {0.2, 0.5, 1.0}, n_steps=20)`。
4. band 內髮絲：取其髮根 texel 的新 latent → `prior.decode(..., roots=原髮根)` 取代原髮絲；band 外不動。
   顏色（base / donor 標籤）沿用原髮絲；需要決定 band 內 texel 該屬於哪顆頭（先用原 keep）。
5. 評估：ELBO 地圖（邊界帶比值）、benchmark 指標（view_mix、slabx、gs；注意 decode 後髮絲是 VAE 重建、較平滑）、渲染。

風險：σ_start 太大會改髮型（band 內變成第三種髮型）；decode 的髮絲比原本平滑 → 邊界帶與周圍質感不同。

### Step 4 — 邊界 transformer + SDS

- 輸入：拼接後 texture（65 ch）+ 邊界 band mask（+ 可選：兩頭各自的 texture）；輸出：band 內的 latent 修正（殘差）。
- loss：`sds_loss`（band 內，σ 取 Step 1 最敏感的範圍）＋ band 外保真（= 0 修正）＋ 修正量正則；
  可選：decode 後在 3D 加 `../nn/diffopt.objective` 的混雜項（讓邊界不交錯）。
- 資料：`$DATA/train_*` 的拼接 case（需在新機器重建 train cases：`build_cases.py 70 --seed 11/12/13 --names_from $DATA/split.json:train`），無需 GT。
- 自監督補充：真實髮型隨機遮一條帶 → 學 inpaint（有 GT）。
- 比較對象：SDEdit baseline、Gaussian mask / hidden-space mask（kung 系列）。

## 4. 注意事項

- 先驗實驗用全髮絲；benchmark 指標原本在 10k 髮絲上算，比較時兩者要分開標註。
- 只用無條件分支（不給 `latents_dict`）。若要條件化，需要融合後的圖像，沒有明確定義。
- `difflocks_prior.py` 內的 igl dtype 與 tensor 轉換修正只在本資料夾，不改 repo 原程式。
- 新方法加入 benchmark 時沿用 `../nn/nn_methods.py` 的註冊方式（opt-in、`bench.py` 執行），不改 `benchmark/`。
