# LOG — fusion mask 精修實驗（benchmark/）

只追加，不改寫已有內容。指標定義見 README「Metrics」；`ret` = min_head_retention（保留髮絲 / 該頭總髮絲，取兩頭較小者），
`gs` = grid_score.mean_all（越低越好），`ret_mask` = 相對 drawn mask 分給該頭的髮絲的保留率。

## 2026-11-20 [0] 同步現況 / 問題定義

- 既有 baseline run：`data/fusion_m2_n50_s0/runs/run`（全髮絲，`tsai_joint_fm` = JOINT_FM_KWARGS：clique、
  `lam=100`、`node_unit="cell"`、`overlap="projection"`、無 balance）。44/50 ok、6 skipped_oom。
  gs 0.170 → 0.048（-56%，win 91%），但 `ret` 平均 1.0 → 0.70，3 個 degenerate（0007、0008、0036 一顆頭 ≈ 0）。
- 逐 case 看：0007、0008、0011、0026、0029、0034、0036、0042 的其中一頭幾乎被吃光（ret ≤ 0.1）；
  0029 FM 後 gs 反而變差（0.117 → 0.175）。多為 mid/long 參與的組合。
- 判斷：`lam` 是絕對值，跟加權交叉數比較；交叉數隨髮絲數平方與兩頭重疊量變化（highlighting Summary 核心結論 3），
  固定 lam=100 在重疊大的 case 幾乎等於 0 → FM 收斂到「全給一顆頭」。這是「不適配某些髮型 / mask」的主因。

## 2026-11-20 [1] 設計：adaptive_fm.py（只在 benchmark 端，不改 joint_contour / img2hair）

- 每 case 自適應 lam：`lam = alpha * W0 / S`（W0 = drawn mask 的加權投影交叉數，S = FM 節點內髮絲數）。
  以 wrapper 包 `joint_contour.core.joint_fm_refine` 實作。
- retention guard：某頭保留 < `min_retention` × mask 分給它的髮絲時，lam ×4 重跑（最多 3 次）。
- 也測既有選項：`balance_tol=0.05`、`overlap="projection_depth"`。
- 同 case 的 membership / projection edges 快取共用（多變體不重算）。
- 方法以 `afm_*` 註冊，只在 `--methods` 指名時執行，預設 run 不變。

## 2026-11-20 [2] 實驗 r1：8 個代表 case，20k 髮絲

- cases：0000（short+long 正常）、0007（mid+long 崩）、0011（long+long 崩）、0023（mid+mid 正常）、
  0029（long+long，FM 變差）、0034（mid+long 崩）、0042（short+long 崩）、0043（short+mid ret 大降）。
- 輸出 `data/dev_adaptive/r1_s20k`（讀 `fusion_m2_n50_s0` 的 cases，不寫入它）。
- 注意：20k 髮絲下 lam=100 的相對強度與全髮絲不同（交叉數約 ×1/25），baseline 也在同設定重跑以便比較。

## 2026-11-22 [3] r1 結果（8 case、20k 髮絲；`python benchmark/dev_table.py data/dev_adaptive/r1_s20k`）

`ret` 以 mask_only 的保留髮絲為分母（= 相對 drawn mask），下表為 8 case 平均；`#ret<.5` = 某頭剩不到一半的 case 數。

| method | gs | ret | ret_min | #ret<.5 | bald |
|---|---|---|---|---|---|
| mask_only | 0.184 | 1.00 | 1.00 | 0 | 0.031 |
| tsai_joint_fm (lam=100) | 0.078 | 0.51 | 0.16 | 5 | 0.038 |
| afm_a0.05 / a0.2 / a0.5 / a1 | 0.005 / 0.009 / 0.038 / 0.059 | 0.18 / 0.20 / 0.34 / 0.35 | ≈0–0.12 | 6–7 | ~0.04 |
| afm_bal0.05 | 0.098 | 0.68 | 0.44 | 2 | 0.037 |
| afm_guard0.5（lam=100 起，×4） | 0.147 | 0.75 | 0.58 | 0 | 0.037 |
| afm_a0.2_guard0.5 | 0.117 | 0.76 | 0.54 | 0 | 0.037 |

觀察：
- 20k 髮絲下 W0/S ≈ 35–50，所以 baseline lam=100 ≈ alpha 2；全髮絲下 W0/S 約 ×5，lam=100 ≈ alpha 0.4 → 全髮絲 baseline 更常崩。
  自適應 lam 的價值主要是「跨髮絲數 / 跨 case 行為一致」，不是 alpha 小本身。
- alpha ≤ 1：gs 極低但幾乎全部 case 某頭被吃光（reward hacking：一頭消失 → 沒有交叉）。不能只看 gs。
- 即使 baseline 在 20k 下，8 case 中 5 個某頭剩 < 50%（0000、0007、0011、0034、0042）。
- guard（×4 跳）能擋崩潰，但跳太大：lam 400 / 1031 時 FM 幾乎不動，gs 回到接近 mask_only（0007：0.126、0011：0.232）。
  → r2：alpha 2 / 5、guard 門檻 0.7、倍率 ×2。
- balance_tol=0.05 介於中間（gs 0.098，ret 0.68），仍有 2 case < 0.5。

## 2026-11-22 [4] r2 結果（同 8 case、20k；結果寫在同一個 `data/dev_adaptive/r1_s20k`）

| method | gs | ret | ret_min | #ret<.5 | bald |
|---|---|---|---|---|---|
| tsai_joint_fm (lam=100) | 0.078 | 0.51 | 0.16 | 5 | 0.038 |
| afm_a1 | 0.059 | 0.35 | 0.12 | 6 | 0.038 |
| afm_a2 | 0.090 | 0.55 | 0.21 | 4 | 0.037 |
| afm_a5 | 0.157 | 0.72 | 0.40 | 1 | 0.036 |
| afm_bal0.05 | 0.098 | 0.68 | 0.44 | 2 | 0.037 |
| afm_a0.2_guard0.5 | 0.117 | 0.76 | 0.54 | 0 | 0.037 |
| afm_a1_guard0.7（×2） | 0.142 | 0.78 | 0.57 | 0 | 0.036 |
| afm_a2_guard0.7（×2） | 0.153 | 0.84 | 0.78 | 0 | 0.036 |

結論：
- 所有 FM 參數變體大致落在同一條 gs ↔ ret 取捨曲線上，**沒有任何設定同時優於 baseline 的 gs 與 ret**。lam 只是取捨旋鈕。
- 重疊大的 case（0007 mid+long、0011 long+long、0034 mid+long）：guard 逐步把 lam 從 80 加到 640 時，
  base 保留率 0.13 → 0.26 → 0.52 → 0.69，gs 也一路回到接近 mask_only（0011：0.263 vs mask 0.233）。
  也就是說只靠「保留 / 藏起」標籤，這些 case 不藏掉大半顆頭就消不掉交叉 —— FM 的本質上限。
- guard 的價值：保證不崩（0/8 case ret < 0.5），可作「安全模式」；代價是重疊大的 case 幾乎不精修。
- alpha ≤ 1 的純自適應 lam 棄用（一頭消失、gs 假低）。
- 下一步（使用者同意，FM 實驗結束後）：不刪髮絲的方法 —— B：rlift 類 strand editing（重疊區 donor 成片抬到 base 上方 + backfill）；
  A：依髮流（向量場）對齊融合邊界。

## 2026-11-22 [5] 方向 B 設計：strand editing（`strand_edit.py`，使用者要求在 FM 實驗後嘗試）

- 不藏髮絲，改成移動：保留 drawn mask（或某個 FM 變體）的所有髮絲，把「上層」頭的髮絲依髮根分成 16×16 texel 的 patch，
  每片用 `strands_editing.stripe_shape.lift` 沿徑向抬到另一顆頭的頭髮上方（只用另一顆頭的 Beer–Lambert 光學深度），
  FTL + `limit_kinks`（30°）後，`backfill` 讓被越過的另一顆頭髮絲往反方向移，維持密度。髮根不動、不刪髮絲。
- 變體：`edit_lift_donor` / `edit_lift_base`（固定上層）、`edit_lift_auto`（另一顆頭在其上方的光學深度中位數較小者 = 原本就比較在外層者當上層）、
  `edit_lift_auto_nobf`（無 backfill）、`edit_guardfm_lift_auto`（先 `afm_a2_guard0.7` 再 lift）。
- 新的 opt-in 指標 `view_mix`（metrics.py）：5 個正交視角的點 z-buffer，每個像素 = 最近點的頭。
  `edge_frac` = 異色相鄰像素對 / 頭髮像素，`block_mix` = 8×8 區塊 4p(1−p) 平均，`donor_vis` = 可見像素中 donor 的比例。
  柱體式 grid score 看不到上下層（抬升後髮絲仍在同一柱體），所以 editing 主要看 view_mix + 渲染；grid score 照常並列。
- runner：`FusionResult.heads`（預設 None）= 編輯後的髮絲；有值時指標改用它，並存 `edited_heads.npz`（float16），
  `render_results` 會用它渲染。None 時行為不變。
- 修正兩個問題：(1) 20k 子集的光學深度要乘 1/frac（每根代表 ~4.75 根），否則 τ 幾乎 0、什麼都不抬；
  (2) 長度 < 1 cm 的髮絲不編輯（lift 的 ramp 0/0 → NaN），任何 NaN 髮絲還原。
- 3000 髮絲的 smoke 不適用：每個 patch < 20 根，lift 直接跳過。
- smoke（case_0011，20k）：donor 抬升最大 18 mm，p95 位移 11.7 mm，backfill 配對 1085 根，約 3 s。
- 觀察：smoke 中 `tsai_joint_fm` 的 view_mix 0.082 → 0.048（edge_frac），donor_vis 0.35 → 0.79（base 大半被藏）。

## 2026-11-22 [6] r3：8 case、20k，加 view_mix（`data/dev_adaptive/r3_edit_s20k`）

r3 結果（8 case 平均；edge = view_mix.edge_frac，blkmix = block_mix，越低越乾淨）：

| method | gs | ret | edge | blkmix | dvis |
|---|---|---|---|---|---|
| mask_only | 0.184 | 1.00 | 0.0625 | 0.119 | 0.286 |
| tsai_joint_fm | 0.078 | 0.51 | 0.0422 | 0.086 | 0.340 |
| afm_a2_guard0.7 | 0.153 | 0.84 | 0.0588 | 0.114 | 0.277 |
| edit_lift_donor | 0.184 | 1.00 | 0.0664 | 0.124 | 0.288 |
| edit_lift_auto / base | 0.184 | 1.00 | 0.0674 | 0.125 | 0.282 |
| edit_lift_auto_nobf | 0.184 | 1.00 | 0.0668 | 0.125 | 0.283 |
| edit_guardfm_lift_auto | 0.153 | 0.84 | 0.0604 | 0.116 | 0.276 |

- 單一上層頭的 lift **全部讓 view_mix 變差**（0.0625 → 0.066–0.067），grid score 不變（預期，柱體看不到上下層）。
- 渲染檢查（`data/dev_adaptive/r3_edit_s20k/cmp_case_0023.png`、`cmp_case_0007.png`，列 = mask_only / tsai / guard FM / edit_lift_auto）：
  view_mix 的高低與渲染一致（case_0023：mask_only 白色 base 髮絲穿過黑色 donor 區；tsai / guard FM 邊界乾淨；
  edit_lift_auto 選了 base 當上層，白髮絲被抬到黑色區上面，更亂）→ view_mix 可作 editing 的指標。
- 原因：「整顆頭當上層」不對。應該是「區域主人在上」：每顆頭在自己的區域蓋住入侵的另一顆頭。

## 2026-11-22 [7] region-owner-on-top lift（`edit_region_lift*`）

- 每顆頭 h：只用「另一顆頭落在 h 區域內的點」（footprint = 最近髮根的 drawn-mask 標籤）算光學深度，h 的 patch 抬到它們上方；
  先 base 後 donor（第二步用更新後的幾何）。backfill 讓入侵者往反方向移。
- case_0023（mid+mid、donor 短捲髮）：edge 0.125 → 0.137（nobf 0.135），仍比 mask_only 差。donor 只抬 0.8 mm：
  露出的白髮絲是在髮根附近就躺在黑髮上的 base 髮絲，lift 的 slope 上限（0.25 × 弧長）讓髮根附近抬不起來。

r4（同 8 case、20k；結果寫在 `data/dev_adaptive/r3_edit_s20k`）：

| method | gs | ret | edge | blkmix |
|---|---|---|---|---|
| mask_only | 0.184 | 1.00 | 0.0625 | 0.119 |
| edit_region_lift | 0.184 | 1.00 | 0.0684 | 0.127 |
| edit_region_lift_nobf | 0.184 | 1.00 | 0.0674 | 0.125 |
| afm_a2_guard0.7 | 0.153 | 0.84 | 0.0588 | 0.114 |
| edit_guardfm_region_lift | 0.153 | 0.84 | 0.0617 | 0.119 |

結論（方向 B 第一輪）：
- 所有 lift 類 editing（單一上層、區域主人在上、有 / 無 backfill、接在 guard FM 後）在 8/8 case 都沒有讓 view_mix 變好，
  平均變差 5–10%；接在 guard FM 後也抵銷了 FM 的改善。→ 棄用 lift 類。
- 原因：融合的雜亂主要不是「深度上交錯」，而是**入侵髮絲在髮根附近就橫躺在對方區域的表層**（分線 / 髮旋附近、長髮垂過邊界）。
  徑向 lift 受 slope 上限，在髮根附近抬不動，抬起的 patch 反而讓邊界更破碎。highlighting 的 rlift 處理的是單頭內的細條紋，條件不同。
- 另發現：view_mix 上 tsai_joint_fm 的 0.042 有一部分來自「一顆頭消失」（dvis 0.34），但 ret 高的 case（0023：ret 0.87、edge 0.048）
  也確實乾淨，FM 的邊界清理本身有效。
- 下一步：側向（沿頭皮切面）把入侵髮絲梳回自己區域 / 依髮流對齊邊界（方向 A）。

## 2026-11-22 [8] 改用 highlighting 的 optical 渲染重新比較（使用者要求）

- 之前的渲染（benchmark 預設）= `npz2blender_kung.py` + `gpu_render.py`：不透明 Principled BSDF、白 / 黑、20k 子集粗髮絲，
  接近 highlighting 的 `raw`，**不是** `strands_editing/pipeline.py` 的 `optical`（全髮絲、metadata 半徑、Chiang hair BSDF melanin、32 bounces、半透明）。
  view_mix 也是不透明 z-buffer。所以 [7]「lift 無效」只在不透明渲染下成立。
- 新增 `render_results.py --optical`（預設關閉）：`render_case_optical` 呼叫 `pipeline.render(preset="optical")`，base = 深棕、donor = 金色（OPTICAL_COLORS），
  輸出 `<method>/render_optical/front_multiview.png`（front / top / left / back）。有 `edited_heads.npz` 時用編輯後的髮絲。
- `common.load_hairstyle` 的 head dict 多一個 `npz_path` 鍵（只加資訊）。`strand_edit._set_density` 改用 metadata 的 hair_profile_radius：
  density = render_density(1, r) / frac（與 pipeline.set_density 相同，再乘子集倍率）。
- r5：全髮絲，4 case（0023 mid+mid、0007 mid+long、0000 short+long、0011 long+long），
  方法 mask_only、tsai_joint_fm、afm_a2_guard0.7、edit_region_lift、edit_guardfm_region_lift，跑完後 optical 渲染。輸出 `data/dev_adaptive/r5_full_optical`。

r5 結果（全髮絲、4 case；`python benchmark/dev_table.py data/dev_adaptive/r5_full_optical`）：

| method | gs | ret | ret_min | edge | blkmix | dvis |
|---|---|---|---|---|---|---|
| mask_only | 0.235 | 1.00 | 1.00 | 0.067 | 0.127 | 0.375 |
| tsai_joint_fm | 0.045 | 0.44 | **0.00** | 0.028 | 0.071 | 0.627 |
| afm_a2_guard0.7 | 0.196 | 0.89 | 0.76 | 0.054 | 0.113 | 0.386 |
| edit_region_lift | 0.235 | 1.00 | 1.00 | 0.075 | 0.139 | 0.372 |
| edit_guardfm_region_lift | 0.197 | 0.89 | 0.76 | 0.058 | 0.119 | 0.386 |

optical 渲染對比（`data/dev_adaptive/r5_full_optical/optical_<case>.png`，欄 = mask_only / tsai / guard FM / region_lift / guard+lift；
base 深棕、donor 金色）：
- case_0023（mid+mid）：tsai 與 guard FM 都讓邊界明顯乾淨、仍照 mask 分區；tsai 稍乾淨。lift 看不出改善。
- case_0007（mid+long）：tsai 把 base（棕色長髮）**整個刪掉**（kept [0, 99816]），變成純 donor 髮型；guard FM 保留背後棕色長髮、邊界略改善。
- case_0011（long+long）：tsai 幾乎換成 donor 的金色直長髮（base 剩 15%），mask 意圖完全失去；guard FM ≈ mask_only（幾乎不動）。
- case_0000（short+long，左右 mask）：tsai 讓 donor（金色短髮）擴大到整個頭頂與前額，形狀大幅偏離 mask；guard FM ≈ mask，小幅改善。
- region_lift：在 optical（半透明）渲染下也沒有可見改善，金色入侵片反而在頂部更明顯（0011 top）。→ 維持棄用。

結論：
- optical 渲染下的判斷與不透明渲染 / view_mix 一致：lift 類無效；FM 是唯一有效的清理方式。
- 全髮絲下 baseline（lam=100 ≈ alpha 0.4）在 4 case 中有 3 個嚴重違反 mask（0007、0011 一頭消失，0000 donor 擴張成另一種髮型）。
  edge / gs 的低分大多來自這種「改髮型」，不是邊界變乾淨。
- `afm_a2_guard0.7` 是目前唯一在所有 case 都維持 mask 意圖的版本，但在重疊大的 case 幾乎不精修。
  → 中間地帶（例如 guard 門檻 0.7、但 alpha 較小 / 只在邊界附近允許移動）值得探索；方向 A（依髮流梳回 / 對齊邊界）仍是下一步。

## 2026-11-22 [9] 比較圖加入原始髮型與 mask（使用者要求）

- 新增 `compare_page.py <run_dir>`：每個 case 一列 = 原始 base｜原始 donor（各自單獨的 optical 渲染，全髮絲）｜2D mask（scalp texture）｜
  頭皮 mask 4 視角（front / top / left / back）｜各方法 optical 渲染 + gs / ret / view_mix。輸出 `<run_dir>/compare/compare_<case>.png`、`index.html`。
- 頭皮 mask 渲染：不用 Blender，matplotlib 正交投影 + painter 排序 + 背面剔除；smplx_base（頸部以上，灰）+ scalp.ply 細分 3 次（~60k 面），
  每個面依 uv 中心的 texel（row = 1 − t、col = s；已驗證 scalp.ply 的 (s, t) = 資料集 root_uv，誤差 0.01）著色：深棕 base / 金色 donor / 灰 = 無髮根區。約 1.2 s / case。
- 原始髮型的 optical 渲染遇到 OptiX misaligned address（0007 donor、0011 donor）→ 自動以 CYCLES_DEVICE=CUDA 重試成功。
- 輸出：`data/dev_adaptive/r5_full_optical/compare/`。
- 2026-11-22 補充（使用者要求）：比較圖改為網格排版（列 = 輸入 / 融合結果 optical / 保留髮絲 mask（頭皮 4 視角）/ 保留髮絲 mask 2D，欄 = 方法），
  並加入每個方法實際保留髮絲的 mask：每 4×4 texel 一格，保留髮根中 donor 的比例（棕 → 金），紅色 = 輸入有髮根但都沒保留（禿）。
  case_0007 可直接看出 baseline 把 base 區整片變成 donor；guard FM 讓 base 區前緣往後縮。注意 mask_only 本身也有少量紅格
  （該格的髮根只屬於 mask 沒分到的那顆頭）。

## 2026-11-22 [10] 方向 A：沿髮流的邊界帶內 FM（`flow_band.py`，使用者要求）

- 動機：baseline FM 可翻任何格子 → 重疊大時整片換頭。交叉來自髮絲沿髮流離開自己區域，所以值得翻的格子在 drawn 邊界附近、且主要沿髮流方向。
- 髮流場（texture 版面）：每根髮絲 root → 沿髮絲 1.5 cm 的點（兩點都以最近髮根對應到 texel），兩顆頭一起在 64×64 上累積無號結構張量、
  高斯 σ=1.5 平滑 → 主方向 f、coherence c。
- 邊界帶：從 drawn mask 的邊界格在 8 連通 texel 圖上跑 Dijkstra，步長代價 |d|·sqrt(k(u·f)² + (u·f⊥)²)，k = 1 − c(1 − 1/a²)；
  a > 1 時沿一致的髮流每格只花 1/a。band = 代價 ≤ budget（texel）。a = 1 為等向帶。
- FM：只有髮根格在 band 內的節點可動（`adaptive_fm.run_variant(node_filter=...)`：在 refine wrapper 內暫時把 `incidence.node_movable` 換成 band、
  `movable_only=True`；`mask_from_node_state` 仍用 movable_only=False，band 外的格子保留 drawn 標籤）。joint_contour 不變。
- 視覺檢查（`data/dev_adaptive/debug_band/bands.png`）：前額髮流沿髮際線橫走、後方向下；帶在髮流平行邊界處窄、橫越邊界處沿髮流延伸。
  flow / band 計算 < 0.3 s。band 佔頭髮區比例：iso16 0.16–0.24、flow32 (a=3) 0.42–0.62。
- 變體：`afb_iso16/32`、`afb_flow16/32`（lam = JOINT_FM_KWARGS 的 100）、`afb_flow16/32_a0.4`（自適應 lam，alpha 0.4 ≈ 全髮絲 baseline 的強度）、
  `afb_flow32_a0.4_g0.7`（再加 guard）。對照：`afm_a0.4`（無 band，同強度）。
- r6：8 case、20k，`data/dev_adaptive/r6_flowband_s20k`。

r6 結果（8 case、20k；edge = view_mix.edge_frac）：

| method | gs | ret | ret_min | #ret<.5 | edge |
|---|---|---|---|---|---|
| mask_only | 0.184 | 1.00 | 1.00 | 0 | 0.0625 |
| tsai_joint_fm | 0.078 | 0.51 | 0.16 | 5 | 0.0422 |
| afm_a0.4（無 band，全髮絲 baseline 強度） | 0.033 | 0.32 | 0.00 | 6 | 0.0319 |
| afm_a2_guard0.7 | 0.153 | 0.84 | 0.78 | 0 | 0.0588 |
| afb_iso16 | 0.153 | 0.82 | 0.72 | 0 | 0.0630 |
| afb_iso32 | 0.134 | 0.70 | 0.56 | 0 | 0.0605 |
| afb_flow16 | 0.143 | 0.76 | 0.56 | 0 | 0.0618 |
| afb_flow16_a0.4 | 0.138 | 0.73 | 0.50 | 0 | 0.0601 |
| afb_flow32 | 0.111 | 0.67 | 0.23 | 2 | 0.0523 |
| afb_flow32_a0.4 | 0.089 | 0.48 | 0.14 | 5 | 0.0438 |
| afb_flow32_a0.4_g0.7 | 0.133 | 0.72 | 0.29 | 1 | 0.0639 |

- band 確實擋住「整片換頭」：同強度下 afm_a0.4 6/8 case 某頭 < 50%，afb_flow16_a0.4 0/8。但 gs↔ret 仍大致沿同一條曲線，沒有明顯支配 guard FM。
- kept mask 圖（`data/dev_adaptive/r6_flowband_s20k/kept_masks.png`）：**band 內的邊界變成破碎的顆粒狀**（0011 iso16 / flow16、0007 flow16 / flow32），
  view_mix 因此上升（0011：0.082 → 0.115–0.118；0007：0.034 → 0.048–0.066）。不受限的 FM 收斂到一大塊（乾淨但改髮型）；
  受限時 FM 停在 band 內逐格翻轉的局部解，projection 目標沒有邊界長度 / 平滑項。
- flow 帶在交叉處沿髮流延伸的效果可見（0023 flow32：donor 區沿前額髮流往兩側延伸，邊界平滑，edge 0.050）。
- band + guard：guard 在 band 內無法救回小區域（0042 short+long：band 32 幾乎蓋滿 donor 的後方小區，lam ×8 後仍被吃）。
- 下一步：band 內 FM 之後做**沿髮流的平滑**（LIC 式沿 streamline 平均 + 門檻），讓邊界與髮流平行、去掉顆粒；也測只對 drawn mask 做沿髮流平滑（不跑 FM）。

## 2026-11-22 [11] r7：band FM + 沿髮流平滑（LIC）（同 8 case、20k，結果在 `data/dev_adaptive/r6_flowband_s20k`）

- `flow_smooth`：band 內每個 texel 取沿 streamline ±L 步（依 f、方向一致）的高斯加權平均（σ = L/2，只計頭髮區樣本），依 coherence 與自身值混合，
  再 σ=1 等向模糊、門檻 0.5；band 外保留。之後依平滑後的 mask 在髮根 texel 重新決定每根髮絲屬於哪顆頭。

| method | gs | ret | ret_min | #ret<.5 | edge |
|---|---|---|---|---|---|
| afm_a2_guard0.7 | 0.153 | 0.84 | 0.78 | 0 | 0.0588 |
| afb_flow32（無平滑） | 0.111 | 0.67 | 0.23 | 2 | 0.0523 |
| afb_iso32_lic12 | 0.137 | 0.82 | 0.67 | 0 | 0.0594 |
| afb_flow16_lic12 | 0.145 | 0.87 | 0.77 | 0 | 0.0598 |
| afb_flow32_lic12 | 0.121 | 0.77 | 0.28 | 1 | 0.0517 |
| afb_flow32_lic24 | 0.114 | 0.77 | 0.24 | 1 | 0.0473 |
| afb_flow32_a0.4_lic12 | 0.113 | 0.57 | 0.19 | 2 | 0.0468 |
| lic_flow32_L12 / L24（只平滑 drawn mask） | 0.184 | 1.00 | 1.00 | 0 | 0.0624 / 0.0621 |

- 平滑後 band FM 的顆粒邊界消失：flow32 → flow32_lic24，gs 不變（0.111 → 0.114）、ret 0.67 → 0.77、view_mix 0.052 → 0.047（非崩潰變體中最好）。
- `afb_flow16_lic12` 大致支配 guard FM（gs 0.145 vs 0.153、ret 0.87 vs 0.84、edge 相近）。
- 只平滑 drawn mask 無效（drawn 邊界是直線，沿髮流平均後門檻幾乎不變）。
- 剩下的失敗：case_0042（short+long、front_back）donor 是後方小區，band 32 幾乎蓋滿 → 被吃（ret 0.24）。
  → r8：band 在每顆頭的區域內最多佔 `cap`（0.5 / 0.3），超過時縮該側的 budget（取該側距離的分位數）。

## 2026-11-22 [12] r8：band cap（同 8 case、20k）

| method | gs | ret | ret_min | #ret<.5 | edge |
|---|---|---|---|---|---|
| afm_a2_guard0.7 | 0.153 | 0.84 | 0.78 | 0 | 0.0588 |
| afb_flow16_lic12 | 0.145 | 0.87 | 0.77 | 0 | 0.0598 |
| afb_flow32_lic24 | 0.114 | 0.77 | 0.24 | 1 | 0.0473 |
| afb_flow32_lic24_cap0.5 | 0.139 | 0.87 | 0.70 | 0 | 0.0590 |
| afb_flow48_lic24_cap0.5 | 0.144 | 0.86 | 0.70 | 0 | 0.0610 |
| **afb_flow32_a0.4_lic24_cap0.5** | **0.131** | 0.83 | 0.67 | 0 | **0.0515** |
| afb_flow32_lic24_cap0.3 | 0.154 | 0.93 | 0.89 | 0 | 0.0589 |

- cap 解決 0042（ret 0.24 → 0.74–0.90）。`afb_flow32_a0.4_lic24_cap0.5` 在 gs 與 view_mix 都優於 guard FM、0/8 崩、ret 稍低；
  最難的 0011（long+long）：0.128 / ret 0.95 / edge 0.072 vs guard 0.263 / 0.84 / 0.113。弱點：0007 edge 0.067（guard 0.034）、0043 ret 0.67。
- 防止在 8 個調參 case 上過擬合：r9 = 20 個 held-out case（20k）比較 mask_only / tsai / guard / 前 3 名，
  之後 r10 = 全髮絲 4 代表 case + optical 渲染。

## 2026-11-22 [13] r9：20 個 held-out case（20k，未用於調參；`data/dev_adaptive/r9_heldout_s20k`）

| method | gs | ret | ret_min | #ret<.5 | edge |
|---|---|---|---|---|---|
| mask_only | 0.161 | 1.00 | 1.00 | 0 | 0.0489 |
| tsai_joint_fm | 0.069 | 0.82 | 0.05 | 1（0036） | 0.0391 |
| **afm_a2_guard0.7** | **0.083** | 0.85 | 0.71 | 0 | **0.0423** |
| afb_flow16_lic12 | 0.138 | 0.91 | 0.60 | 0 | 0.0483 |
| afb_flow32_lic24_cap0.5 | 0.127 | 0.92 | 0.71 | 0 | 0.0500 |
| afb_flow32_a0.4_lic24_cap0.5 | 0.116 | 0.86 | 0.60 | 0 | 0.0499 |
| afb_flow32_lic24_cap0.3 | 0.141 | 0.94 | 0.82 | 0 | 0.0528 |

- **排名反轉**：held-out 上 guard FM 最好（gs 接近 baseline、edge 0.042、0 崩），band 系列 gs 只降 15–30%、view_mix 幾乎沒比 mask_only 好（部分 case 更差，如 0030：0.006 → 0.039–0.049）。
- 原因：8 個調參 case 是刻意挑的「崩潰 / 重疊大」case，band 在那裡有利；一般 case 中 band 限制了 FM 的改善範圍，LIC 門檻又引入新的可見邊界。
  → band 系列在 8 case 上的優勢是選樣造成的，不能單獨取代 guard FM。
- guard FM 與 band 互補（guard 在重疊大的 case 幾乎不動，band 在那裡有效）→ r11：每個 case 同時算兩者，
  只保留 ret ≥ 0.7 的候選，選加權投影交叉數（FM 自己的目標，不用評估指標）最少者：`afs_guard_or_band`、`afs_guard_or_band3`（多 afb_flow16_lic12）。

## 2026-11-22 [14] 資料遺失與重建（23:37）

- 23:37（使用者 commit `1122` 之後兩分鐘）`benchmark/data/` 底下所有 .json / .log / .html / .md / .csv 被刪除（原因不明；png / npz 都在）：
  包括 `fusion_m2_n50_s0` 的 benchmark.json、50 個 case.json、原始 50 case run 的 result.json / summary / report，以及本實驗所有 run 的 result.json / log。
  執行中的 r10 / r11 中斷。程式碼與 LOG 在 commit 中，未受影響。
- 重新 build（現行 / 10-26 版 builder、seed 0、有無 --min_hair_length 0.12）都**無法重現**原 case（髮型與 mask 不同；原 benchmark 建於 10-15，早於第一個 commit）。
- 經使用者同意，改為重建 case.json：髮型名稱與 mask 類型取自原 run 的 log（`benchmark/nohup.out`，50 個 case 都有）、mask 用保留下來的 `mask_1.png`、
  長度重新計算；每顆頭的全髮絲數與 log 一致（50/50）。**case seed 無法還原**，改為 1000 + index（`seed_reconstructed: true`），
  所以 `--max_strands` 子集與原本不同；全髮絲結果不受影響。驗證：case_0023 mask_only 全髮絲 gs 0.3540、kept [75503, 21858]，與遺失前完全相同。
- 原始 50 case run（`runs/run`）：由保留的 maps.npz keep mask 重算所有指標寫回 result.json（`restored` 欄位；FM objectives / 時間等 method_info 遺失；
  6 個 tsai skipped_oom case 依先前 summary 標回 skipped_oom）。抽查 case_0000 / 0001 與遺失前一致。
- 本實驗 20k 結果（r1、r3、r6、r9）的數字只保留在本 LOG；其 keep mask 對應舊 seed，無法再重算。之後的 20k 比較全部在新 seed 下重跑：
  r12 = 28 case（8 調參 + 20 held-out）× mask_only / tsai / guard / afb_flow16_lic12 / afb_flow32_a0.4_lic24_cap0.5 / afs_guard_or_band(3)。
  r10（全髮絲 4 case + optical）重跑並加入 `afs_guard_or_band`。

## 2026-11-23 [15] r12（28 case、20k、新 seed）與 r10（全髮絲 4 case + optical）

r12（`data/dev_adaptive/r12_all28_s20k`）：

| method | 28 case gs / ret / ret_min / #<.5 / edge | 調參 8：gs / ret / edge | held-out 20：gs / ret / edge |
|---|---|---|---|
| mask_only | 0.167 / 1.00 / 1.00 / 0 / 0.054 | 0.184 / 1.00 / 0.062 | 0.160 / 1.00 / 0.050 |
| tsai_joint_fm | 0.072 / 0.72 / 0.06 / 7 / 0.041 | 0.077 / 0.51 / 0.042 | 0.070 / 0.81 / 0.040 |
| afm_a2_guard0.7 | 0.104 / 0.85 / 0.72 / 0 / 0.050 | 0.150 / 0.83 / 0.059 | 0.086 / 0.85 / 0.047 |
| afb_flow16_lic12 | 0.141 / 0.90 / 0.58 / 0 / 0.051 | – | – |
| afb_flow32_a0.4_lic24_cap0.5 | 0.118 / 0.86 / 0.61 / 0 / 0.050 | 0.128 / 0.83 / 0.050 | 0.114 / 0.87 / 0.050 |
| afs_guard_or_band(3) | 0.103 / 0.84 / 0.72 / 0 / 0.051 | 0.152 / 0.82 / 0.065 | 0.083 / 0.85 / 0.045 |

- 新 seed 下重現 r9 的結論：held-out 上 guard FM 最好；調參 8 case 上 band 最好。
- `afs_guard_or_band`：25/28 選了 guard。以「加權投影交叉數」挑選**挑錯**：調參 8 case edge 0.065（比兩個候選都差），
  全髮絲 0011 選了 guard（0.292）而非 band（0.147），0007 選了最差的 band。交叉數會偏好藏較多髮絲 / 改動較大的候選 → 棄用此選擇準則。

r10（`data/dev_adaptive/r5_full_optical`，全髮絲 4 個重疊大的代表 case，比較圖 `compare/compare_<case>.png`）：

| method | gs | ret | ret_min | edge |
|---|---|---|---|---|
| mask_only | 0.235 | 1.00 | 1.00 | 0.067 |
| tsai_joint_fm | 0.045 | 0.44 | 0.00 | 0.028 |
| afm_a2_guard0.7 | 0.196 | 0.89 | 0.76 | 0.054 |
| afb_flow16_lic12 | 0.172 | 0.89 | 0.82 | 0.049 |
| afb_flow32_a0.4_lic24_cap0.5 | 0.164 | 0.85 | 0.73 | 0.051 |
| afb_flow32_lic24_cap0.3 | 0.189 | 0.90 | 0.74 | 0.054 |

- optical 渲染（0011 long+long）：band 兩版的邊界沿髮流、乾淨，兩顆頭都保留（ret 0.97–0.98，gs 0.257 → 0.147–0.192）；guard ≈ 不動（0.292）；baseline 換成 donor 髮型。
  0007：afb_flow32_a0.4 變差（0.231 / edge 0.077），flow16_lic12 較穩（0.153 / 0.92）。
- 結論：沒有單一方法支配。guard FM = 一般 case 的穩健選擇；band 系列 = 重疊大時唯一能精修又不換髮型的方法。下一步需要一個好的「何時用 band」準則
  （候選：兩頭重疊度 / overlap_matrix.density、guard 的 lam 被拉高的倍數 = 衝突嚴重程度），而不是交叉數。
- 2026-11-23：原始 run 的 run_config.json 依 session 早先印出的內容補回，summarize 重新產生 summary / report：All cases mask_only gs 0.1695 ± 0.098、tsai 44 ok / 6 failed，與遺失前的 summary 一致。

## 2026-11-23 12:00 checkpoint

- 無執行中的實驗；文件（LOG / SUMMARY / checkpoints）已同步到 [15]。
- 目前最佳：`afm_a2_guard0.7`（Promising，穩健、0/28 崩）；重疊大的髮型用方向 A `afb_flow16_lic12` / `afb_flow32_a0.4_lic24_cap0.5`（Experimental）。
- 已棄用：純自適應 lam（alpha ≤ 1）、rlift 類 strand editing、無平滑 / 無 cap 的 band、以交叉數挑選的 `afs_*`。
- 未完成：何時改用 band 的準則；band 參數在全髮絲 held-out 上的驗證；回寫 `img2hair_tsai_1.py`（需加預設關閉的 flag，並經使用者同意）。

## 2026-11-23 [16] 少量髮絲能否取代完整投影？（使用者提問）

- DiffLocks 的 guide strands 只有 135 根 × 32 點，且推論流程沒有 → 不可行；「少量髮絲」= 解碼髮絲的隨機子集。
- 子集測試（4 個全髮絲 case，guard FM 在子集上跑、mask 套回全部髮絲再評分；腳本 `/tmp/claude-1000/subset/subset_test.py`）：
  - 子集 + 256 解析度：子集沒抽到的格子 FM 決定不了、保留原標籤 → 翻與沒翻的格子交錯，常比不精修還差（0023：20k → gs 0.357 > mask 0.354）。
  - 子集 + 粗解析度（2k@32 / 5k@64）：與全髮絲 FM 的改動 IoU 約 0.6–0.7、0.2–2 s，但品質差（0023：0.29 vs 全髮絲 0.17）——粗格放大後階梯邊界、子集交叉估計雜訊大。
  - 結論：少量髮絲能快速找出「大致該改哪裡」，但不能直接取代完整投影。
- → 兩段式（`two_stage.py`）：(1) 5k / 2k 髮絲、64 / 32 粗格跑 guard FM（約 1–3 s）；(2) 全部髮絲，從 stage-1 mask 出發，
  只讓「stage-1 改動區 + 邊界」擴張 W texel 的帶內節點可動，且**只在帶內髮絲經過的柱體建投影交叉**（帶外兩根髮絲之間的交叉不影響任何移動 → 最佳解不變，省掉大部分圖）。
  lam = 2·W0/S_band，guard 0.7。adaptive_fm 加兩個 hook：`edge_rows`（限制建邊柱體，guard 重跑時共用）、`lam_scope="movable"`。
- smoke（0023、20k）：guard FM 108 s、gs 0.128、kept [13662, 3872]；`ts_5k_r64_w8` 7 s、gs 0.118、kept [13850, 4822]（更好且保留更多）。
- r13：全髮絲 4 case（0023 / 0007 / 0000 / 0011）跑 `ts_5k_r64_w8`、`ts_5k_r64_w16`、`ts_2k_r32_w16`，之後 optical 渲染。

## 2026-11-23 [17] r13 兩段式（全髮絲 4 case）與時間分析；membership 加速

r13（全髮絲 4 case）：

| method | gs | ret | ret_min | edge | 時間 |
|---|---|---|---|---|---|
| mask_only | 0.235 | 1.00 | 1.00 | 0.067 | – |
| tsai_joint_fm | 0.045 | 0.44 | 0.00 | 0.028 | 410 s |
| afm_a2_guard0.7 | 0.196 | 0.89 | 0.76 | 0.054 | 412 s |
| ts_5k_r64_w8 | 0.198 | 0.86 | 0.72 | 0.061 | 321 s（含 membership） |
| ts_5k_r64_w16 | 0.176 | 0.84 | 0.68 | 0.049 | 33 s + membership |
| ts_2k_r32_w16 | 0.114 | 0.86 | 0.73 | 0.041 | 33 s + membership |

- 時間拆解：全髮絲每 case 的 membership（256 格、strand → column）約 **290 s**，舊方法 410 s 中只有約 120 s 是交叉圖 + FM；
  兩段式把後者降到約 33 s（第 1 段 2 s、限制柱體的建邊 15–28 s、FM 約 8 s），記憶體峰值 17 → 11.7 GB（整個 case 進程，僅供參考）。
- 注意：ts_2k_r32_w16 在 0000 gs 0.065、kept [58338, 40725]，與 baseline 幾乎相同（donor 從 28.5k 擴張到 40.7k）——guard 只擋減少、不擋增加，需看渲染確認是否「改髮型」。
- membership 加速（`fast_membership.py`，不改 joint_contour）：256 格分 16×16 → 4×4 兩層區塊，每層用區塊平均座標系 + 容許值
  （區塊大小 + 高度 × 法向傾角 + 1 mm）過濾候選點，最細層用原本的精確測試。結果與原函式比對：20k 髮絲 29.2 → 5.2 s（差 1 個 entry，邊界浮點），
  **全髮絲 144 → 28 s，完全相同（0 差異）**。以 `fast_membership=True`（預設關閉）接入 `adaptive_fm.run_variant`；新方法
  `afm_a2_guard0.7_fast`（結果應與 guard 完全相同）、`ts_5k_r64_w16_fast`、`ts_2k_r32_w16_fast`。
- r14：全髮絲 4 case 跑上述三者（兩段式先跑，量單獨使用的總時間）。

## 2026-11-23 [18] r14：fast membership 驗證、從 0 跑的時間、兩段式 vs guard

- `afm_a2_guard0.7_fast` 的 keep mask 在 4/4 個全髮絲 case 與 `afm_a2_guard0.7` **逐根完全相同**；兩段式 fast / 非 fast 的 gs 也完全相同。
- 從 0 跑的時間（全髮絲，不含讀檔 / 評估）：舊 guard / tsai 約 410–460 s（其中交叉圖 + FM 只約 47 s，其餘幾乎都是 membership 等前置）；
  **guard_fast 約 100 s（結果相同，約 4–5×）**；兩段式 fast 約 80–95 s（第 1 段 4–7 s + membership 56 s + 限制柱體投影 15–28 s + FM）。
  → 主要加速來自 fast membership；兩段式相對 guard_fast 只再省 5–20 s。
- 兩段式是否逼近 guard：guard 有在精修的 case 逼近良好（0023：改動髮絲 IoU 0.92 / 0.95、98.8–99.2% 髮絲決定相同；0007：IoU 0.66–0.67、97.4%）；
  guard 因重疊大幾乎不動的 case（0000 改動 2.5%、0011 0.1%）兩段式改得多（7–22%）——第 1 段的保留率檢查在子集 / 粗格上、第 2 段 lam 只以帶內正規化，保守程度不同。
- 渲染（`compare/compare_case_0000.png`）：`ts_2k_r32_w16` 在 0000 重現 baseline 的「donor 擴張到頭頂、改髮型」（kept mask 幾乎與 tsai 相同）；
  `ts_5k_r64_w16` 的 kept mask 出現顆粒狀區域。0011 上 ts_2k 較好（gs 0.292 → 0.131、ret 0.99）。
- 結論：要「更快得到 guard 的結果」用 `afm_a2_guard0.7_fast`；兩段式不是 guard 的近似，而是較積極、較不穩定的變體（有時更好、有時改髮型），暫列 Experimental。
  guard 只擋「某頭減少」、不擋「某頭擴張」是 0000 失敗的共同原因 → 可加「donor / base 各自不超過 mask 的 1 + δ」的雙向 guard。

## 2026-11-23 [19] 合併效果與髮流的關係（使用者提問；r12 的 28 case、20k，數據 `/tmp/claude-1000/flow_dependence.json`）

- 兩個便宜、FM 前就能算的特徵：
  - intrusion = 依 drawn mask，保留髮絲的點落在「另一顆頭區域」上方的比例（整根髮絲的走向 × 長度，footprint = 最近髮根）；
  - flow_across = drawn 邊界上 |髮流 · 邊界法向| × coherence 的平均（髮根附近 1.5 cm 的局部髮流，即 flow_band 的方向場）。
- Spearman（28 case）：intrusion vs mask_only gs 0.46（p 0.013）；**intrusion vs guard gs 0.78（p < 0.001）**；**intrusion vs tsai 保留率 −0.66（p < 0.001）**；
  flow_across vs mask_only gs 0.10（p 0.60，無關）；flow_across vs intrusion 0.33。
- 解讀：難度由「整根髮絲跨進對方區域多少」決定（長髮下垂、髮流與 mask 的相對方向），**不是髮根附近的局部髮流方向**；
  intrusion 高時 baseline 傾向吃掉一顆頭、guard 傾向不動（例如 0007 0.22、0011 0.18、0019 0.29、0034 0.24、0042 0.43）。
  但不是單調門檻（0002 0.20、0015 0.26、0016 0.27 guard 仍有效）。
- intrusion 不需要交叉圖即可算 → 可作為「何時改用 band / 較積極方法」準則的候選；也提示方向 A 的方向場應改用整根髮絲的走向（而非髮根 1.5 cm）。

## 2026-11-23 [20] 深度分離規則（slab）與側向梳理（使用者要求）

- 使用者規則：同一投影柱體內、但高度（沿格子法向）明顯不同的兩種髮絲可以允許（分層，不算交錯）。
- `slab_edges.py`：每個柱體投影一次，線段依高度分到厚 dz、半重疊的 slab，只有同一 slab 內投影相交才算交叉（權重 = 有此交叉的柱體數）。
  只用 joint_contour 現有的 numba 函式，不改它。驗證：dz 涵蓋整個高度時與 `joint_projection_edges` **完全相同**。
  case_0011（20k）：dz 2 / 1 / 0.5 cm 時只剩 63% / 40% / 24% 的交叉對（權重 56% / 33% / 18%）——長髮的「交叉」多半是上下分層；建邊時間約 2×。
- 接入 guard FM：`afm_a2_guard0.7_slab1` / `_slab2`（fast membership + slab dz 1 / 2 cm，自適應 lam 以 slab 圖的 W0 正規化）。
- 新的 opt-in 指標 `slabx`：在 32×32 分析格上算保留髮絲的投影交叉權重 `proj_w` 與同高度交叉 `slab_w`（dz 1 cm），千為單位。
- `strand_comb.py`（側向梳理）：FM 後，保留髮絲上 footprint 落在對方區域、且 3 mm 體素鄰域內確實有對方頭髮（深度分離者不動）的點，
  沿切面往自己區域（侵蝕 2 texel 後的最近 texel 的頭皮點）移動，上限 d_max；沿髮絲平滑、髮根漸入、同頭 8 個鄰近髮絲平均，FTL 保長、轉折 ≤ 30°。
  變體 `comb10/20_guard`、`comb10_guard_refit`（梳完再跑一次 FM）、`comb10/20_slab1`。
- smoke（0011、20k）：guard（原規則）slab_w 929 → 222、view_mix 0.082 → 0.121（變差）、lam 被拉到 317；
  **guard + slab 1 cm：slab_w 79（-91%）、view_mix 0.070、ret 0.86–0.91、lam 60 不需升高**；梳理再降 10–20%（slab1：79 → 70），位移平均 3–6 mm、p99 9–11 mm。
- r15：28 case（20k）× mask_only / tsai / guard_fast / guard_slab1 / guard_slab2 / comb10/20 × guard/slab1，指標含 view_mix、slabx。

r15 結果（`data/dev_adaptive/r15_slab_comb_s20k`，28 case、20k；same-h = slabx 的同高度交叉 / mask_only；p99 = 被移動髮絲位移的 p99）：

| method | 28：gs / ret_min / #<.5 / vm / same-h | 調參 8：gs / vm / same-h | held-out 20：gs / vm / same-h | p99 |
|---|---|---|---|---|
| mask_only | 0.167 / 1.00 / 0 / 0.0535 / 1.00 | 0.184 / 0.0620 / 1.00 | 0.160 / 0.0500 / 1.00 | – |
| tsai_joint_fm | 0.072 / 0.06 / **7** / 0.0408 / 0.20 | 0.077 / 0.0419 / 0.07 | 0.070 / 0.0404 / 0.26 | – |
| guard_fast | 0.104 / 0.72 / 0 / 0.0501 / 0.21 | 0.150 / 0.0588 / 0.35 | 0.086 / 0.0466 / 0.16 | – |
| guard_slab1 | 0.115 / 0.72 / 0 / 0.0463 / 0.14 | 0.144 / 0.0497 / 0.19 | 0.103 / 0.0449 / 0.12 | – |
| guard_slab2 | 0.108 / 0.72 / 0 / 0.0461 / 0.17 | 0.138 / 0.0525 / 0.23 | 0.096 / 0.0436 / 0.14 | – |
| comb10_guard | 0.095 / 0.72 / 0 / 0.0463 / 0.15 | 0.141 / 0.0524 / 0.25 | 0.077 / 0.0438 / 0.11 | 8.4 mm |
| comb20_guard | 0.089 / 0.72 / 0 / 0.0427 / 0.12 | 0.135 / 0.0477 / 0.20 | **0.071** / **0.0407** / 0.09 | 14.8 mm |
| comb10_slab1 | 0.106 / 0.72 / 0 / 0.0433 / 0.11 | 0.136 / 0.0447 / 0.14 | 0.094 / 0.0427 / 0.09 | 8.0 mm |
| **comb20_slab1** | 0.101 / 0.72 / 0 / **0.0413** / **0.09** | 0.131 / **0.0411** / 0.12 | 0.090 / 0.0414 / **0.08** | 14.1 mm |

- 全部梳理 / slab 版本 0/28 崩（baseline 7/28）。**梳理 2 cm 讓 view_mix 達到 baseline 水準（0.041）而不吃頭**；調參 8（重疊大）上 comb20_slab1 0.0411 < baseline 0.0419。
- slab 規則降低「同高度交叉」（guard 0.21 → 0.14），view_mix 也降；但 grid score 與全投影交叉上升（不看高度，預期）。
- 梳理的代價是位移：1 cm 版 p99 約 8 mm、2 cm 版約 14–15 mm（重疊大的 case 17–19 mm）。外觀需 optical 渲染確認（reward hacking 風險）。
- r16：全髮絲 4 case 跑 guard_slab1、comb10_slab1、comb20_slab1、comb20_guard + optical 渲染。

## 2026-11-23 [21] r16：全髮絲 4 case + optical（slab / 梳理）

| method | gs | ret_min | view_mix | 0000 / 0007 / 0011 / 0023 的 view_mix |
|---|---|---|---|---|
| mask_only | 0.235 | 1.00 | 0.067 | 0.051 / 0.028 / 0.065 / 0.122 |
| tsai_joint_fm | 0.045 | 0.00 | 0.028 | 0.034 / 0（base 消失）/ 0.033 / 0.044 |
| afm_a2_guard0.7 | 0.196 | 0.76 | 0.054 | 0.061 / 0.024 / 0.072 / 0.059 |
| guard_slab1 | 0.191 | 0.75 | 0.045 | 0.052 / 0.026 / 0.046 / 0.054 |
| comb10_slab1 | 0.179 | 0.75 | 0.041 | 0.046 / 0.024 / 0.042 / 0.052 |
| **comb20_slab1** | **0.173** | 0.75 | **0.040** | 0.042 / 0.023 / 0.042 / 0.052 |
| comb20_guard | 0.180 | 0.76 | 0.045 | 0.048 / 0.021 / 0.062 / 0.051 |

- 渲染（`r5_full_optical/compare/compare_<case>.png`，欄 mask_only / tsai / guard / guard_slab1 / comb20_slab1）：0011（long+long）guard 幾乎不動、
  slab 版邊界清楚且兩頭都保留（ret 0.86），梳理後交界處的金色細絲更少；0023 各版邊界乾淨。整頭視角下**未見豎起、成團或明顯彎折**（2 cm 上限、p99 約 14 mm）。
  近距離細節尚未檢查。
- 結論：`comb20_slab1` 在 28 case（20k）與 4 個全髮絲重疊大 case 上 view_mix 都是不崩方法中最好（≈ baseline 水準），且 0 崩 → 升為 Promising。

## 2026-11-23 [22] 演算法移到 `inference/fusion_refine/`（使用者要求）

- 套件：`adaptive.py`（自適應 lam、guard、node / column 限制、slab、fast membership；`refine_joint_fm`）、`fast_membership.py`、`slab_edges.py`、
  `comb.py`、`flow.py`、`two_stage.py`（helpers）、`presets.py`（`refine(heads, labels, "comb20_slab1" | ...)`，給推論腳本用）、`checks.py`（與 joint_contour 原函式的等價檢查）。
  只依賴 `{"positions", "root_uv"}` + scalp-texture mask，不依賴 benchmark；joint_contour 不變。
- benchmark 的 `adaptive_fm.py`、`flow_band.py`、`two_stage.py`、`strand_comb.py` 改為只註冊方法的 wrapper（方法名不變）；`benchmark/fast_membership.py`、`slab_edges.py` 移除，
  `metrics.slabx` 改 import 套件。`strand_edit.py`（已棄用）留在 benchmark。
- 驗證：移動前後在 case_0011 / 0023（20k）跑 10 個方法（mask_only、guard、guard_fast、guard_slab1、comb20_slab1、comb10_guard_refit、afb×2、ts_fast、afs），
  maps.npz（keep mask）、edited_heads.npz、全部指標 **20/20 完全相同**；`presets.refine(..., "comb20_slab1")` 的 keep 與位置也與 benchmark 版相同。
  `python -m fusion_refine.checks membership` 重現 fast membership 的等價檢查。
- `img2hair_tsai_1.py` 尚未改動（接入需預設關閉的 flag）。
