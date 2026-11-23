# checkpoints — 可用的融合方法版本

每個版本 = `fusion_methods.METHODS` 中的一個方法名（`--methods` 指名）；演算法本體在 `inference/fusion_refine/`（`presets.py` 有 `guard`、`guard_slab1`、`comb10_slab1`、`comb20_slab1`）。數字標明 run 與髮絲數；
`gs` = grid_score.mean_all，`ret` = 兩頭中較低的「保留髮絲 / drawn mask 分給它的髮絲」。

## Promising

- `comb20_slab1`（`adaptive_fm.py` slab + `strand_comb.py`）：guard FM（自適應 lam alpha 2、retention guard 0.7、fast membership）以**深度規則**計交叉
  （同柱體但高度差 > ~1 cm 不算，`slab_edges.py`），之後把仍與對方頭髮同高度交錯的入侵髮絲沿頭皮切面梳回自己區域（上限 2 cm、FTL 保長、轉折 ≤ 30°）。
  28 case（20k）：view_mix 0.0535 → **0.0413**（baseline 0.0408 但 7/28 崩；guard 0.0501），同高度交叉 → mask 的 9%，0/28 崩，位移 p99 約 14 mm。
  全髮絲 4 個重疊大 case：view_mix 0.067 → **0.040**（guard 0.054），gs 0.173，ret_min 0.75；optical 渲染未見豎起 / 成團。
  缺點：會移動髮絲（平均數 mm、p99 約 14–19 mm）；grid score 不看高度，所以 gs 改善小於 view_mix。變體 `comb10_slab1`（位移 p99 約 8 mm）view_mix 0.0433 / 0.041。

- `afm_a2_guard0.7`（`adaptive_fm.py`）：自適應 lam（lam = 2·W0/S，跨髮絲數 / case 一致）+ retention guard（某頭 < 0.7 × mask 分給它的髮絲時 lam ×2 重跑，≤ 3 次）。
  r12（28 case、20k）：gs 0.167 → 0.104（-38%）、ret 0.85、ret_min 0.72、0/28 崩（baseline：0.072、ret 0.72、7/28 某頭 < 50%）、view_mix 0.054 → 0.050。
  held-out 20 case：gs 0.086（baseline 0.070）、ret 0.85、view_mix 0.047。全髮絲 4 個重疊大 case：gs 0.196、ret_min 0.76，渲染上 4/4 守住 mask。
  優點：簡單、從不吃掉一顆頭、一般 case 接近 baseline 的精修量。缺點：重疊大時 lam 被拉高後幾乎不精修（0011：0.292 vs mask 0.257）。

## Experimental

- `afm_a2_guard0.7_fast`：與 `afm_a2_guard0.7` 逐根相同的結果，membership 改用 `fast_membership.py`（全髮絲 144 → 28 s / 頭，0 差異）。
  全髮絲從 0 跑約 100 s（原版 410–460 s）。驗證完成後可取代原版作為 Promising 的實作。
- `ts_5k_r64_w16(_fast)`、`ts_2k_r32_w16(_fast)`（`two_stage.py`）：子集粗格 guard FM → 全髮絲帶內 FM，只在帶內髮絲經過的柱體建投影。
  從 0 跑約 80–95 s。guard 有在精修的 case 與之高度一致（0023 IoU 0.92–0.95）；重疊大的 case 改動多得多：0011 更好（ts_2k gs 0.131、ret 0.99），
  0000 ts_2k 重現 baseline 的 donor 擴張（改髮型）、ts_5k 出現顆粒狀 mask。不穩定，需雙向 guard（擋擴張）後再評估。

- `afb_flow16_lic12`、`afb_flow32_a0.4_lic24_cap0.5`（`flow_band.py`，方向 A）：FM 只在「沿髮流的邊界帶」內翻格子（Dijkstra，沿髮流代價 1/3），
  之後沿髮流做 LIC 平滑去掉顆粒狀邊界；cap0.5 = 帶子最多佔每顆頭區域的一半。
  全髮絲 4 個重疊大 case：gs 0.172 / 0.164、ret_min 0.82 / 0.73、view_mix 0.049 / 0.051（guard 0.196 / 0.76 / 0.054）；0011 渲染邊界沿髮流、兩頭都保留。
  r12 28 case（20k）：gs 0.141 / 0.118、0/28 崩，但 held-out 一般 case 上精修量明顯少於 guard（0.114 vs 0.086）。
  適合重疊大的髮型；需要一個「何時改用 band」的準則。

- `afm_a2_guard0.7`（`adaptive_fm.py`）：自適應 lam（alpha=2，lam = 2·W0/S）+ retention guard（某頭 < 0.7 時 lam ×2 重跑，≤ 3 次）。
  r1（8 case、20k）：gs 0.153、ret 0.84、ret_min 0.78、0/8 case 崩（baseline 0.078 / 0.51 / 0.16 / 5）。
  優點：任何髮型 / mask 都不會吃掉一顆頭、跨髮絲數行為一致；缺點：重疊大的 case 幾乎不精修（gs ≈ mask_only）。適合「安全模式」。
  r5（全髮絲 4 case + optical 渲染）：gs 0.235 → 0.196、ret 0.89、ret_min 0.76、view_mix edge 0.067 → 0.054；
  渲染上 4/4 case 都維持 mask 分區（baseline 3/4 嚴重違反：一頭消失或 donor 擴張成另一種髮型）。
- `afm_a0.2_guard0.5`：同上，alpha=0.2、門檻 0.5、×4。gs 0.117、ret 0.76、ret_min 0.54、0/8 崩。比上者更積極。
- `afm_bal0.05`：baseline + balance_tol=0.05。gs 0.098、ret 0.68、2/8 case ret < 0.5。
- `afm_a2`、`afm_a5`：純自適應 lam。a2 ≈ baseline 在 20k 的行為（0.090 / 0.55）；a5 0.157 / 0.72。
- `afm_depth`、`afm_depth_a0.2`（overlap="projection_depth"）：已實作，未測。

新指標 `view_mix`（`--metrics ...,view_mix`，LOG [5]）：5 視角 z-buffer 的異色相鄰像素比例，與渲染一致，可評估移動髮絲的方法。
r3 8 case / 20k：mask_only 0.0625、tsai_joint_fm 0.0422、afm_a2_guard0.7 0.0588。

## Baseline

- `tsai_joint_fm`：`img2hair_tsai_1.JOINT_FM_KWARGS`（clique、lam=100、cell 節點、projection overlap）。
  r5（全髮絲 4 case + optical 渲染）：gs 0.045、edge 0.028 最低，但 ret_min 0.00；0007 / 0011 一頭消失、0000 donor 擴張，低分多來自改髮型。
  全髮絲 50 case：gs 0.170 → 0.048（win 91%）；ret 平均 0.70，部分 mid/long case 一頭被吃光。
  20k 髮絲 8 代表 case：gs 0.078、ret 0.51、5/8 case 某頭 < 50%。
- `mask_only`：不精修。
- `tsai2_joint_fm_trim`（`--tsai2`）：baseline FM + trim。

## 已棄用

- `afm_a0.05`、`afm_a0.2`、`afm_a0.5`、`afm_a1`：純自適應 lam、alpha ≤ 1。gs 0.005–0.059 但 6–7/8 case 一頭被吃光（reward hacking）。
- `afm_guard0.5`（lam=100 起、×4）：被 `afm_a2_guard0.7` 取代（ret 較低、gs 較高）。
- `edit_lift_donor`、`edit_lift_base`、`edit_lift_auto`、`edit_lift_auto_nobf`、`edit_guardfm_lift_auto`、`edit_region_lift`、
  `edit_region_lift_nobf`、`edit_guardfm_region_lift`（`strand_edit.py`，rlift 類徑向抬升 + backfill，LOG [5]–[7]）：
  8/8 case view_mix 不降反升（mask_only 0.0625 → 0.066–0.068；guard FM 0.0588 → 0.0617）。融合的雜亂來自髮根附近橫躺在對方區域表層的入侵髮絲，
  徑向 lift 受 slope 上限抬不動。r5 的 optical（半透明）渲染也無可見改善。程式保留供參考。
- `afs_guard_or_band`、`afs_guard_or_band3`（`flow_band.py`）：每 case 以加權投影交叉數在 guard / band 間挑選。25/28 選 guard，且挑錯
  （全髮絲 0011 選 guard、0007 選最差的 band；調參 8 case view_mix 比兩個候選都差）。交叉數不是好的選擇準則。
- `afb_iso*`、`afb_flow16/32`（無平滑）、`afb_*_a0.4`（無 cap）、`afb_flow32_a0.4_g0.7`、`lic_flow32_*`（只平滑 drawn mask）：
  被有平滑 + cap 的版本取代（無平滑時帶內邊界成顆粒，view_mix 上升；無 cap 時小區域被吃；只平滑 drawn mask 無效果）。

