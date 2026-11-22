# checkpoints — 可用的融合方法版本

每個版本 = `fusion_methods.METHODS` 中的一個方法名（`--methods` 指名）。數字標明 run 與髮絲數；
`gs` = grid_score.mean_all，`ret` = 兩頭中較低的「保留髮絲 / drawn mask 分給它的髮絲」。

## Promising

（尚無：FM 參數變體都在 baseline 的同一條 gs ↔ ret 取捨曲線上，見 LOG [4]）

## Experimental

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
