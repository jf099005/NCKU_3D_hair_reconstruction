# checkpoints — fusion_refine/nn 可用版本

SOTA 標記（2026-11-24）：品質 = `dopt_fm_mix10`、全 NN / 速度 = `dnet_dn1_k1`；完整對照表見 `SUMMARY.md`「重要結果」。

每個版本 = 一個 checkpoint 目錄（`$DATA/runs/<name>/`，含 config.json）+ `nn_methods.py` 中的 benchmark 方法名。
數字標明 case 集與髮絲數；`gs` = grid_score.mean_all，`vm` = view_mix，`ret_min` = 兩頭中較低的保留率。

## Promising

- **[SOTA 全 NN] `dnet_dn1_k1`**、`dnet_dn1_fm`（`$DATA/runs/dn1`；`disp_net.py` + `train_dnet.py` + `prep_dnet.py`，方法 `nn_methods.run_dnet`）：把 `dopt` 攤銷成逐根位移 NN，
  無標註，以 diffopt 目標（w_mix 10）訓練；輸入 = 每根 32 點的自身 / 對方密度、對方密度梯度、法向、相對位置 + 髮絲描述子，一步同頭鄰居平均，2 pass。0.31 M 參數。
  val 16 case：k1 版 gs 0.045、vm **0.0426**、slab_w 5.8，**全 NN 約 1.9 s / case**；FM 版 vm 0.0485、slab_w 4.9（梳理 0.0498 / 10.5）。gap_to_opt 0.059。
  test 12 case（長髮）：k1 版 vm 0.0447（梳理 0.0450）但 gs 0.110、slab_w 33.8；FM 版 gs 0.093、vm 0.0457、slab_w 6.6。
  優點：不需迭代即可得到最佳化 ~94% 的效果、全 NN 管線比 FM + 梳理快約 20×；缺點：移動髮絲數較多（約 6k，多為小幅）、長髮上受 k1 選髮絲限制；未渲染、未在全髮絲驗證。

- **[SOTA 品質] `dopt_fm_mix10`**、`dopt_k1_mix10`（`diffopt.py` + `nn_methods.run_dopt`）：keep（NN k1 / FM guard_slab1）之後，逐 case 以 Adam 直接最小化可微混雜目標
  （每頭保留點處對方頭的 3 mm 體素模糊密度）+ 位移 / 徑向 / 平滑 / 鄰居一致 / 段長正則，逐根 8 knots 位移，FTL + kink ≤ 30°。不需 teacher。
  val 16 case（10k）：k1 版 gs 0.043、vm 0.0428、slab_w 3.7、ret_min 0.73，約 4.5 s / case；FM 版 gs 0.046、vm 0.0488、slab_w 4.0（梳理 teacher：0.049 / 0.0498 / 10.5）。
  test 12 case（mid / long）：FM 版 gs 0.091、vm 0.0437、slab_w 4.2（梳理 0.091 / 0.0450 / 14.1）；k1 版 gs 0.110、vm 0.0458、slab_w 22.5、ret_min 0.84（k1 選髮絲是長髮上的瓶頸）。
  渲染（val 0000 / 0002 / 0007，LOG [10]）：外觀自然，無豎起 / 彎折。
  優點：首個在全部指標勝過手工梳理的位移方法；位移 p99 ≈ 10 mm（梳理 13 mm）。缺點：目標與 slabx 相近（slab_w 改善部分由定義而來）；kink 限制觸發較多；
  未在全髮絲驗證；仍是逐 case 最佳化，尚未攤銷成 NN。變體 `dopt_*`（w_mix 1）較保守（p99 5–6 mm，vm 0.0451 / 0.0509）。

## Experimental

- `nn_k1__nodisp`（`$DATA/runs/k1`，`keep_model.py` + `train_keep.py`；benchmark 方法名經 `bench.py` 註冊）：逐根 keep 分類器取代 FM，
  texture context（64×64 特徵）+ 28 維髮絲描述子 → flip logit，threshold log(w_flip)，ret < 0.7 時退回 drawn split。teacher = comb20_slab1 的 keep（10k 髮絲）。
  val 16 case（10k）：gs 0.058（FM 0.059）、ret 0.86 / ret_min 0.73（FM 0.86 / 0.78）、view_mix 0.0480（FM 0.0545、teacher 0.0498）、slab_w 30.5（FM 19.5）、0.65 s / case（FM 37 s）。
  test 12 case（全 mid / long，較難）：gs 0.121（FM 0.103）、ret 0.90 / ret_min 0.84（FM 0.84 / 0.70）、vm 0.0506（FM 0.0510）、slab_w 148（FM 30）→ 長髮大重疊時弱於 FM、偏保守。
  渲染（val 4 case，LOG [5]）：一般 case 邊界乾淨（0007 優於 FM），捲髮 / 大重疊（0002）漏掉入侵髮絲；未見「丟表層髮絲」的 hacking。
  優點：約 40–57× 快、不需交叉圖（RAM）、保留率高；缺點：長髮 / 大重疊時同高度交叉明顯多於 FM；未在全髮絲驗證。變體 `nn_k1_t3__nodisp`（較保守：ret 0.90、vm 0.0466、gs 0.067）。

- `d2kf_regress`（`$DATA/runs/d2kf_regress`，`train.py --mode regress --feat_keep`，width 32、wd 1e-2、dropout 0.1）：keep 之後的特徵 → 64×64 位移場（regress）。
  方法：`nn_k1__d2kf_regress`（全 NN）、`nn_teacher__d2kf_regress`（FM keep + NN 位移）。val 16 case：k1 + d2kf vm 0.0480 → 0.0471、slab_w 30.5 → 27.1；
  FM + d2kf vm 0.0545 → 0.0540、slab_w 19.5 → 16.7（梳理：0.0498 / 10.5）。改善只有梳理的 10–30%；位移被平均化（動 9k 根、p99 3–4 mm vs 梳理 1.5k 根、p99 10 mm）。
  比 d1 好但仍遠不如手工梳理；保留作為兩段式的參考。

## Baseline

- `comb20_slab1`（benchmark，`inference/fusion_refine/presets.py`）：guard FM + slab 1 cm 深度規則 + 2 cm 側向梳理。v0 的 teacher。
  數字見 `benchmark/checkpoints.md`；本專案 val / test 的數字見 LOG。
- `afm_a2_guard0.7_slab1`：只有 FM（不梳理）。
- `mask_only`：不精修。

## 已棄用

- `d1_flow`、`d1_regress`（scalp texture 位移場，flow matching / regress，以 drawn split 的特徵預測 teacher 梳理位移）：監督誤差不勝過「不動」，
  benchmark 上 vm 改善 ≤ 0.0003（梳理 teacher 改善 0.005）。LOG [3]、[4]。位移需要不同的條件（refined keep 之後的特徵）或目標。
- `d2kf_flow`（兩段式特徵的 flow matching 位移）：監督誤差 rel 1.43（比不動差），benchmark 與 d2kf_regress 相當或較差（vm 0.0476，K=4 挑選 0.0470）。LOG [7]。
