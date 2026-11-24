# SUMMARY — fusion_refine/nn

逐步紀錄見 `LOG.md`，可用版本見 `checkpoints.md`。

## 目前狀態

- 目標：NN 取代 FM（選髮絲）與梳理（位移）。資料 train 210 / val 16 / test 12 cases（10k 髮絲）。
- 位移：以可微混雜目標無標註訓練的逐根 NN `dn1`（Promising）拿到逐 case 最佳化約 94% 的效果；全 NN 管線（k1 + dn1）1.9 s / case，val vm 0.0426（所有方法最低）（LOG [12]）。
- 直接最佳化 `dopt_*_mix10`（Promising）在 val / test 都勝過手工梳理（LOG [9]–[11]）。
- 選髮絲：NN `k1` 在長髮（test）偏保守、gs 較差，是目前的主要瓶頸（LOG [6]、[11]、[12]）。
- 沒有執行中的實驗。
- 下一步：dn1 渲染；選髮絲也改以可微目標訓練（soft keep）或長髮加權；全髮絲驗證。

## 重要結果

### SOTA（2026-11-24，10k 髮絲；val 16 case、test 12 case（全 mid / long））

| 標記 | 方法 | 選髮絲 / 位移 | val gs / vm / slab_w | test gs / vm / slab_w | ret_min（val / test） | 秒 / case |
|---|---|---|---|---|---|---|
| **SOTA（品質）** | `dopt_fm_mix10` | FM / 可微目標直接最佳化 | 0.046 / 0.0488 / 4.0 | **0.091 / 0.0437 / 4.2** | 0.78 / 0.70 | 41 |
| **SOTA（全 NN）** | `dnet_dn1_k1` | NN k1 / NN dn1 | 0.045 / **0.0426** / 5.8 | 0.110 / 0.0447 / 33.8 | 0.73 / 0.84 | **1.9** |
| 次佳（NN 位移） | `dnet_dn1_fm` | FM / NN dn1 | 0.048 / 0.0485 / 4.9 | 0.093 / 0.0457 / 6.6 | 0.78 / 0.70 | 37 |
| 參考 | `dopt_k1_mix10` | NN k1 / 直接最佳化 | **0.043** / 0.0428 / **3.7** | 0.110 / 0.0458 / 22.5 | 0.73 / 0.84 | 4.5 |
| 原 Promising | `comb20_slab1` | FM / 手工梳理 | 0.049 / 0.0498 / 10.5 | 0.091 / 0.0450 / 14.1 | 0.78 / 0.70 | 39 |
| 參考 | `afm_a2_guard0.7_slab1` | FM / – | 0.059 / 0.0545 / 19.5 | 0.103 / 0.0510 / 30.5 | 0.78 / 0.70 | 37 |
| 參考 | `nn_k1__nodisp` | NN k1 / – | 0.058 / 0.0480 / 30.5 | 0.121 / 0.0506 / 148.2 | 0.73 / 0.84 | 0.65 |
| 參考 | `mask_only` | – / – | 0.120 / 0.0586 / 120.6 | 0.182 / 0.0587 / 313.9 | 1.00 / 1.00 | 0 |

- SOTA（品質）= 長髮（test）上 gs 不輸梳理、vm 與同高度交叉最低且不吃頭的方法；SOTA（全 NN）= 不需 FM / 交叉圖的最快組合，val vm 最低。
- Baseline `tsai_joint_fm` 未在全部 case 上跑；在渲染的 5 case 中長髮會改髮型（test 0010 donor 只剩 2 根）（LOG [13]）。
- 比較圖：`$DATA/compare/all_{val,test}_case_*.jpg`（LOG [13]）。
- 限制：只在 10k 髮絲驗證；slab_w 與可微目標定義相近（vm、gs 為獨立指標）；全 NN 在長髮上受 k1 選髮絲限制。

### 其他結論


- [val / test] 無標註攤銷位移 NN dn1：val vm 0.0426（全 NN，1.9 s / case）；test FM keep 下與梳理相當、slab_w 6.6 vs 14.1（LOG [12]）。

- [test 12 case] 直接最佳化位移在長髮上同樣勝過手工梳理（FM keep：vm 0.0437 vs 0.0450、slab_w 4.2 vs 14.1）（LOG [11]）。

- [val 16 case] 可微混雜目標直接最佳化勝過手工梳理：FM keep 下 vm 0.0488 vs 0.0498、slab_w 4.0 vs 10.5；NN keep 下 vm 0.0428（LOG [9]）。

- [val 16 case] 兩段式 NN 位移 d2kf_regress：FM keep 下 vm 0.0545 → 0.0540（梳理 0.0498）、slab_w 19.5 → 16.7（梳理 10.5）（LOG [7]）。

- [test 12 case, mid/long] NN 選髮絲弱於 FM：gs 0.121 vs 0.103、slab_w 148 vs 30、ret_min 0.84 vs 0.70（LOG [6]）。

- [val 16 case, 10k] NN 選髮絲 k1：gs 0.058（FM 0.059）、vm 0.0480（FM 0.0545）、slab_w 30.5（FM 19.5）、0.65 s vs 37 s（LOG [4]）。
- [val 16 case] NN 位移 d1 無效（vm 改善 ≤ 0.0003）。

- [smoke 2 case] 位移場 64×64 表示幾乎無損（oracle vm 0.048 / 0.061 vs teacher 0.047 / 0.058）；mask 壓到 64 / 128 則失去 FM 的效果 → keep 要逐根預測（LOG [2]）。
