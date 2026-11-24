# LOG — fusion_refine/nn（append only）

## 2026-11-24 [1] 開始：NN 取代 FM（選髮絲）與梳理（位移）

- 使用者要求：以 flow matching / diffusion 等 NN 取代 `fusion_refine/` 的 FM 演算法與小幅位移方法。
- 環境：此機器（RTX 4070 12 GB、31 GB RAM）的資料集 / highlighting / venv 在外接碟 `hair_work/`（189 個髮型，無 guide_strands.npz、無 DiffLocks checkpoint）。
  → `run.sh` 設路徑；`build_cases.py` 以 full_strands 算髮長（快取 `$DATA/lengths.json`）。
- 參考：benchmark 的 Promising `comb20_slab1`（guard FM + slab 深度規則 + 2 cm 梳理）在 10k 髮絲時 FM 約 30–45 s / case、梳理 1–2 s；
  全髮絲約 100 s + 交叉圖 RAM 約 12 GB。NN 的潛在好處：速度 / 記憶體、可多次取樣再挑、可學到超越手工規則的位移。
- 監督來源的選擇（沒有「正確融合」的 GT）：
  - v0 = teacher 蒸餾 + 取樣挑選：以 `comb20_slab1` 的 refined mask 與位移當目標，訓練 scalp texture 上的條件 flow matching；
    推論時取 K 個樣本，以便宜的「同高度混雜」代理分數 + 保留率挑一個（不直接用評估指標 view_mix，避免 reward hacking）。
    上限受 teacher 限制，但可驗證「texture 空間的 NN 能否表達 FM + 梳理的決策」。
  - 之後的候選：自監督的髮絲先驗（真實髮型的局部一致性，denoise / SDEdit 式位移）、以可微代理目標直接訓練（amortized optimization）。
- 資料：split（base style，seed 0）train 126 / val 47 / test 16 個髮型；cases（`--min_hair_length 0.03`）train 3×70、val 16、test 12。
  teacher：`comb20_slab1` @ 10k 髮絲（val / test 另跑 mask_only、guard_slab1，指標含 view_mix、slabx）。

## 2026-11-24 [2] 管線 smoke 與表示法的上限（oracle）

- 管線：`features.py`（64×64 scalp texture 特徵 46 ch：drawn label、region、座標、每頭密度 / 形狀（root 座標系）/ 長度、
  footprint（最近髮根的 texel）上的高度分層密度、平均方向、與對方頭髮的 3 mm 體素混雜）、`prep.py`（teacher 結果 → npz）、
  `model.py`（條件 UNet，flow matching 或 regress）、`nn_methods.py`（benchmark 方法）、`bench.py`（註冊後呼叫 run_benchmark；
  child worker 也經 bench.py）。位移套回：texel 場在髮根雙線性取樣 → 8 個 knot 間線性插值 → 髮根 smoothstep → FTL 保長 → kink ≤ 30°（同 comb.py）。
- teacher 位移很稀疏：smoke 6 case 只有 1–14% 的 texel 移動 > 0.5 mm，中位 1–2 mm、最大 ~15 mm；refined mask 只改 0.3–3% 的 texel，且多為零散單一 texel。
  → 位移的正規化尺度從 2 cm 改 5 mm。
- oracle（teacher 的輸出壓到 R×R 再套回；smoke case_0003 / 0005，10k 髮絲）：

| 方法 | gs 0003 / 0005 | view_mix 0003 / 0005 |
|---|---|---|
| mask_only | 0.019 / 0.110 | 0.049 / 0.084 |
| afm_a2_guard0.7_slab1（FM） | 0.0075 / 0.064 | 0.051 / 0.065 |
| comb20_slab1（teacher） | 0.0019 / 0.059 | 0.047 / 0.058 |
| oracle mask@64 + disp@64 | 0.015 / 0.104 | 0.052 / 0.078 |
| oracle mask@128 + disp@128 | 0.017 / 0.106 | 0.051 / 0.079 |
| oracle mask@256 + disp@256 | 0.0049 / 0.061 | 0.048 / 0.060 |
| oracle 原 mask + disp@64 | 0.0039 / 0.060 | 0.048 / 0.061 |

- 結論：**位移場在 64×64 表示幾乎無損**；**選髮絲必須是逐根（256 texel ≈ 每根）的決策**，pooled 到 64 / 128 幾乎失去 FM 的全部效果。
  → 設計改為兩個模型：
  - keep：`keep_model.py` 逐根分類（texture context CNN 在髮根取樣 + 髮絲描述子 28 維：形狀、長度、侵入比例、與對方混雜比例（分高度 / 髮根附近）、
    髮根處對方密度、到 drawn 邊界距離）→ flip（keep ≠ drawn split）logit，BCE、flip 加權。smoke 上 teacher 丟掉的髮絲 mix 0.5–1.0 vs 保留的 0.01–0.2，描述子有鑑別力。
  - disp：`model.py` 的 UNet 只生成位移場（24 ch），flow matching（N(0, I) → 位移，Euler 16 步）；`_k<K>` 取 K 個樣本以 mixing 代理挑一個，`_dz1` 把 < 1 mm 的取樣抖動歸零。
  - 「選髮絲」部分不是生成模型（逐根分類）；若之後需要多樣性，候選是不同 threshold 或 Bernoulli 取樣再挑。
- ablation 方法：`nn_teacher__<disp>`（teacher FM 的 keep + NN 位移）、`nn_<keep>__nodisp`。

## 2026-11-24 [3] 第一輪訓練（train 210 case × 2 subset = 420、val 16 case × 2 = 32）

- teacher 資料完成（3 shard 並行，約 95 min）。val teacher10k（16 case）：mask_only gs 0.120 / vm 0.0586 / slab_w 120.6；
  guard_slab1 0.059 / 0.0545 / 19.5（ret_min 0.78）；comb20_slab1 0.049 / 0.0498 / 10.5。teacher FM 每 case 約 37 s。
- keep `k1`（keep_model，60 ep，w_flip 10，約 3 s / ep）：val flip F1 0.54（prec 0.39、rec 0.86，threshold logit 0；預測 flips 為真實的 2.2×）。
  推論 threshold 預設改 log(w_flip)（還原 flip 加權的先驗）；另註冊 t0 / t1 / t3 變體。
- disp（UNet 11.6 M，300 ep）：以「取樣位移 vs teacher」MSE / 「不動」MSE（rel）衡量，**兩者都沒有明顯勝過不動**：
  - `d1_regress`：最佳 rel 0.87（ep 34），之後 train loss 0.0017 vs val 0.042 → 嚴重過擬合。
  - `d1_flow`：最終 rel 1.09（最佳 1.08）→ 取樣誤差 ≥ 不動。
  - 推測原因：teacher 位移依賴 FM 之後的 refined keep 與局部混雜（特徵以 drawn split 計算），且資料只有 210 case、位移極稀疏。
- 下一步：val benchmark（`val_eval1.log`）比較 nn_k1 各 threshold、nn_k1 + d1、nn_teacher + d1_regress；依結果決定位移改用
  refined keep 的特徵（兩段式：先 keep 再算特徵）或改自監督目標。

## 2026-11-24 [4] val 16 case benchmark（10k 髮絲，`val/runs/teacher10k`）

| method | 選髮絲 / 位移 | gs | ret（各 case 較低頭的平均）/ ret_min | view_mix | slab_w | 秒 / case |
|---|---|---|---|---|---|---|
| mask_only | – / – | 0.120 | 1.00 / 1.00 | 0.0586 | 120.6 | 0 |
| afm_a2_guard0.7_slab1 | FM / – | 0.059 | 0.86 / 0.78 | 0.0545 | 19.5 | 37 |
| comb20_slab1（teacher） | FM / 梳理 | 0.049 | 0.86 / 0.78 | 0.0498 | 10.5 | 37 + 2 |
| **nn_k1__nodisp** | NN / – | 0.058 | 0.86 / 0.73 | **0.0480** | 30.5 | **0.65** |
| nn_k1_t0__nodisp | NN（thr 0）/ – | 0.098 | 0.91 / 0.76 | 0.0699 | 64.2 | 0.5 |
| nn_k1_t3__nodisp | NN（thr 3）/ – | 0.067 | 0.90 / 0.80 | 0.0466 | 39.7 | 0.5 |
| nn_k1__d1_regress | NN / NN regress | 0.056 | 0.86 / 0.73 | 0.0477 | 29.6 | 1.2 |
| nn_k1__d1_flow_dz1 | NN / NN flow | 0.057 | 0.86 / 0.73 | 0.0478 | 30.3 | 1.1 |
| nn_teacher__d1_regress | FM / NN regress | 0.058 | 0.86 / 0.78 | 0.0544 | 18.8 | 29 |

- **NN 選髮絲（k1，thr = log 10）≈ FM 的 gs（0.058 vs 0.059）、保留率相同、view_mix 更低（0.048 vs FM 0.0545、teacher 0.0498），約 57× 快**；
  但同高度交叉 slab_w 較高（30 vs 19.5），ret_min 0.73 < 0.78。view_mix 低於 teacher 而 slab_w 較高 → 可能是丟掉的髮絲較偏可見表層，需渲染確認（reward hacking 風險）。
- thr 0（多翻 2×）明顯變差：多出的 flip 包含把對方區域的髮絲「加回來」，反而增加混雜。thr 3 較保守（ret 0.90、vm 0.0466、gs 0.067）。
- **NN 位移無效**：k1 + d1 只讓 vm 0.0480 → 0.0477；FM + d1_regress 0.0545 → 0.0544（梳理 teacher 為 0.0498）。與 LOG [3] 的監督誤差一致 → d1 系列棄用。

## 2026-11-24 [5] val 渲染（Blender 5.2 `/home/jf099005-u/blender-5.2.0-linux-x64/blender`，CYCLES_DEVICE=CUDA，raw preset，base 白 / donor 黑）

- 比較圖：`$DATA/compare/val_case_{0000,0002,0007,0011}.png`（列 mask_only / guard_slab1 / comb20_slab1 / nn_k1__nodisp）。
  渲染指令：`PYTHONPATH=hair_work:benchmark:inference:repo venv/python benchmark/render_results.py <run> --cases ... --blender_path <上面路徑>`（約 2 min / case × 4 方法）。
- 0000（NN vm 最低但 slab_w 66 vs FM 5）：NN 讓 donor 區保持接近 drawn mask（donor 保留 2490 vs FM 2014，FM 把黑區往後縮）；
  但頭頂前方仍有少量黑髮絲壓在白色瀏海上 → slab_w 高的原因。**不是「丟表層髮絲」的 reward hacking**，而是邊界保留了少數入侵髮絲。
- 0002（捲髮、NN 較差）：NN 右側黑白交錯、後頸白色捲髮未清除；FM 明顯乾淨。→ 捲髮 / 大重疊時 NN 漏掉入侵髮絲。
- 0007：NN 頂視邊界乾淨（斜線分界），FM / comb 在白區留下零散黑髮絲 → NN 較好。
- 0011：三者相近，邊界都乾淨。
- 結論：NN 選髮絲在一般 case 可取代 FM，失敗模式是捲髮 / 大重疊時漏掉入侵髮絲（recall 不足的那一側）。

## 2026-11-24 [6] test 12 case（全部 mid / long，10k 髮絲，`test/runs/teacher10k`）

| method | gs | ret / ret_min | view_mix | slab_w | 秒 |
|---|---|---|---|---|---|
| mask_only | 0.182 | 1.00 / 1.00 | 0.0587 | 313.9 | 0 |
| afm_a2_guard0.7_slab1（FM） | 0.103 | 0.84 / 0.70 | 0.0510 | 30.5 | 38 |
| comb20_slab1（teacher） | 0.091 | 0.84 / 0.70 | 0.0450 | 14.1 | 38 + 4 |
| nn_k1__nodisp | 0.121 | 0.90 / 0.84 | 0.0506 | 148.2 | 1.0 |
| nn_k1_t3__nodisp | 0.136 | 0.91 / 0.77 | 0.0529 | 171.9 | 0.6 |

- test 比 val 難（mask_only slab_w 314 vs 121）。**NN 在 test 上弱於 FM**：gs 0.121 vs 0.103、同高度交叉 148 vs 30（mask 314），view_mix 持平（0.0506 vs 0.0510）；
  保留率較高（0.90 / 0.84 vs 0.84 / 0.70）——NN 翻的髮絲比 FM 少，偏保守。
- 與 val 渲染 0002 的失敗模式一致：重疊大 / 長髮時漏掉入侵髮絲。訓練資料以 short 為主（train 髮型 short 77 / mid 30 / long 19），長髮大重疊的 case 少。
- k1 在 checkpoints 維持 Experimental；改進方向：長髮 / 大重疊 case 加權或增加資料、recall 導向的 threshold（依 case 的侵入比例調整）。
- 同時開始 [7]：兩段式位移（`prep.py --keep_feat`：特徵以 teacher 保留的髮絲計算；推論時以預測的 keep 計算）。
  `d2kf_regress`（width 32、wd 1e-2、輸入 channel dropout 0.1、100 ep）、`d2kf_flow`（同上、150 ep）。

## 2026-11-24 [7] 兩段式位移 d2kf（特徵以保留後的髮絲計算）val 16 case

- 監督誤差（rel，1 = 不動）：`d2kf_regress` 0.72（ep 49；d1_regress 0.87）；`d2kf_flow` 1.43（更差）。

| method | 選髮絲 / 位移 | gs | vm | slab_w | 移動髮絲數 / p99（base / donor） |
|---|---|---|---|---|---|
| afm_a2_guard0.7_slab1 | FM / – | 0.059 | 0.0545 | 19.5 | – |
| comb20_slab1（teacher） | FM / 梳理 | 0.049 | 0.0498 | 10.5 | 1470 / 4.7 / 10.3 mm |
| nn_teacher__d2kf_regress | FM / NN | 0.057 | 0.0540 | 16.7 | 9239 / 3.1 / 4.4 mm |
| nn_k1__nodisp | NN / – | 0.058 | 0.0480 | 30.5 | – |
| nn_k1__d2kf_regress | NN / NN | 0.055 | 0.0471 | 27.1 | 9133 / 3.3 / 4.6 mm |
| nn_k1__d2kf_flow_dz1 | NN / NN flow | 0.057 | 0.0476 | 29.7 | 9133 / 5.1 / 6.4 mm |
| nn_k1__d2kf_flow_k4 | NN / NN flow ×4 挑選 | 0.056 | 0.0470 | 28.7 | – |

- 兩段式有幫助但很小：FM keep 下 d2kf_regress 只拿到梳理約 10% 的 vm 改善（0.0545 → 0.0540 vs 0.0498）、約 30% 的 slab_w 改善（19.5 → 16.7 vs 10.5）。
- 失敗型態：regress 把稀疏的大位移「平均掉」——移動 9k 根、幅度小（p99 3–4 mm），teacher 只移動 1.5k 根、p99 到 10 mm。
  flow 本應避免平均化，但 210 case 下取樣誤差大於訊號（rel 1.43）；K=4 挑選只帶來 0.0006。
- 判斷：以 texture 平均位移場蒸餾梳理，受限於 (a) 資料量、(b) 梳理位移逐根稀疏而 texel 平均把它抹平。
  候選：逐根位移（與 keep 同樣在髮根取樣 context + 髮絲描述子，只對「混雜」髮絲預測）、或不蒸餾而直接以可微的混雜代理訓練。

## 2026-11-24 [8] 方向 3：可微混雜目標（不蒸餾梳理）——先驗證目標本身（逐 case 直接最佳化）

- 使用者選擇方向 3。先不訓練 NN：`diffopt.py` 對每個 case 直接以 Adam 最佳化逐根 knot 位移（8 knots × 3，髮根 smoothstep，套回同 apply），
  確認目標能改善 benchmark 再攤銷（amortize）成 NN。
- 目標：mixing = 每頭保留點處「對方頭」的密度（點三線性 splat 到 3 mm 體素、σ = 1 voxel 高斯模糊；兩頭都可動、梯度雙向），以自身密度正規化；
  正則：|D|²（單位 5 mm，w 0.02）、徑向分量²（w 0.2，避免豎起）、沿髮絲二階差分（0.5）、同頭 8 鄰居一致（0.5）、32 點折線段長變化（5）、位移軟上限 2 cm。
  之後 FTL 保長、kink ≤ 30°。
- smoke（val case_0000，FM keep，10k）：mix 0.0135 → 0.0019（w_mix 1）/ 0.0004（w_mix 10），150 iter 約 3–4 s；
  移動 2.1k / 2.6k 根、p99 6.5 / 10.4 mm（與梳理的稀疏型態相似，不像 d2kf 的平均化）。
- 方法：`dopt_{fm,k1}{,_mix3,_mix10}`（keep 來源 FM / NN k1 + 直接最佳化）。val 16 case 評估中。

## 2026-11-24 [9] 直接最佳化 val 16 case 結果

- 修正：`img2hair_tsai_1` import 時全域關閉 autograd → `diffopt.optimize` 內以 `torch.enable_grad()` 包住（dopt_fm 第一次全部失敗，已重跑）。

| method | 選髮絲 / 位移 | gs | ret / ret_min | vm | slab_w | 移動根數 / p99 max / kink 限制 | 秒 / case |
|---|---|---|---|---|---|---|---|
| afm_a2_guard0.7_slab1 | FM / – | 0.059 | 0.86 / 0.78 | 0.0545 | 19.5 | – | 37 |
| comb20_slab1（手工梳理） | FM / 梳理 | 0.049 | 0.86 / 0.78 | 0.0498 | 10.5 | 1470 / 13.3 mm / 17 | 37 + 2 |
| dopt_fm | FM / 最佳化 w_mix 1 | 0.052 | 0.86 / 0.78 | 0.0509 | 8.1 | 1984 / 5.2 mm / 68 | 37 + 4 |
| **dopt_fm_mix10** | FM / 最佳化 w_mix 10 | **0.046** | 0.86 / 0.78 | **0.0488** | **4.0** | 2397 / 9.4 mm / 170 | 37 + 4 |
| nn_k1__nodisp | NN / – | 0.058 | 0.86 / 0.73 | 0.0480 | 30.5 | – | 0.65 |
| dopt_k1 | NN / 最佳化 w_mix 1 | 0.049 | 0.86 / 0.73 | 0.0451 | 10.6 | 2092 / 6.1 mm / 74 | 5 |
| **dopt_k1_mix10** | NN / 最佳化 w_mix 10 | **0.043** | 0.86 / 0.73 | **0.0428** | **3.7** | 2523 / 10.3 mm / 164 | **4.5** |

- **可微混雜目標有效，且勝過手工梳理**：同為 FM keep 時，w_mix 10 在 gs / vm / slab_w 都優於梳理（0.046 / 0.0488 / 4.0 vs 0.049 / 0.0498 / 10.5），p99 位移更小（9.4 vs 13.3 mm）。
- **全 NN keep + 直接最佳化（dopt_k1_mix10）是目前 val 最好的組合**：gs 0.043、vm 0.0428、slab_w 3.7，每 case 約 4.5 s（不需 FM / 交叉圖）。
- 注意：(a) 目標本身就是 3D 鄰近混雜，與 slabx 定義相近 → slab_w 的改善部分是「照指標最佳化」；vm（渲染式）與 gs 是獨立指標，也都改善。
  (b) kink 限制觸發次數較多（164–170 vs 17）→ 髮絲可能有轉折，需渲染檢查。(c) 尚未在 test（長髮）驗證。
- 這一步還是逐 case 最佳化（Adam 150 iter，GPU 約 4 s），不是 NN；下一步是渲染 + test，再把它攤銷成 NN（以此目標無監督訓練逐根位移模型）。

## 2026-11-24 [10] dopt 渲染（val 0000 / 0002 / 0007，Blender 5.2 raw preset）

- 比較圖：`$DATA/compare/dopt_case_{0000,0002,0007}.png`（列 mask_only / comb20_slab1 / dopt_fm_mix10 / nn_k1__nodisp / dopt_k1_mix10）。
- 整頭視角下**未見豎起、成團或明顯彎折**（kink 限制觸發較多，但在此解析度看不出）。
- 0000：dopt_k1_mix10 左視邊界成一條乾淨的直線，頭頂前方壓在白瀏海上的黑髮絲比 nn_k1 少；dopt_fm_mix10 與梳理相近。
- 0002（捲髮，最難）：dopt_k1_mix10 後視比 nn_k1 乾淨，但右視仍有一束黑髮絲橫過白區 → 位移上限（2 cm）內移不開的入侵，需要在選髮絲端處理。
- 0007：dopt_k1_mix10 頂視邊界乾淨；dopt_fm_mix10 與梳理一樣在白區留下零散黑髮絲（來自 FM 的 keep，位移無法移除）。
- 結論：直接最佳化的位移外觀自然，視覺上與指標一致；剩下的主要缺陷來自選髮絲（捲髮 / 大重疊漏掉入侵髮絲）。

## 2026-11-24 [11] dopt 在 test 12 case（mid / long）

| method | gs | ret / ret_min | vm | slab_w | 秒 |
|---|---|---|---|---|---|
| mask_only | 0.182 | 1.00 / 1.00 | 0.0587 | 313.9 | 0 |
| afm_a2_guard0.7_slab1（FM） | 0.103 | 0.84 / 0.70 | 0.0510 | 30.5 | 38 |
| comb20_slab1（手工梳理） | 0.091 | 0.84 / 0.70 | 0.0450 | 14.1 | 42 |
| **dopt_fm_mix10** | 0.091 | 0.84 / 0.70 | **0.0437** | **4.2** | 42 |
| nn_k1__nodisp | 0.121 | 0.90 / 0.84 | 0.0506 | 148.2 | 1.0 |
| dopt_k1_mix10 | 0.110 | 0.90 / 0.84 | 0.0458 | 22.5 | 6.6 |

- 長髮上同樣成立：FM keep 時直接最佳化 ≥ 梳理（gs 相同、vm 0.0437 vs 0.0450、slab_w 4.2 vs 14.1）。
- NN keep + 最佳化：slab_w 148 → 22.5、vm 0.0506 → 0.0458，但 gs 0.110 仍不如 FM 系（0.091），保留率較高（ret_min 0.84 vs 0.70）。
  → 長髮上的差距主要在選髮絲（k1 偏保守），位移部分已不是瓶頸。
- 開始 [12]：把目標攤銷成逐根 NN（`disp_net.py` / `train_dnet.py` / `prep_dnet.py`）：
  輸入 = 每根 32 點上的自身 / 對方密度、對方密度梯度、法向、相對髮根位置 + 28 維髮絲描述子；MLP + 一步同頭鄰居平均；輸出 8 knots 位移；
  2 次 pass（第二次以移動後的點重算密度）。訓練損失 = diffopt.objective（w_mix 10），無標註；keep = teacher FM 的 keep。`dn1`：30 ep、hidden 256。
  diffopt 重構為 prepare / objective / optimize（val case_0000 的最佳化歷程與重構前完全相同）。

## 2026-11-24 [12] 攤銷位移 NN `dn1`（無標註，以 diffopt 目標訓練）

- 訓練：train 210 case（keep = teacher FM），30 ep × 約 20 s，0.31 M 參數。val 目標 0.168（不動）→ 0.043；mix 0.0168 → 0.0026；
  與逐 case 最佳化的差距 gap_to_opt 0.059（= 拿到最佳化 94% 的改善）。
- 修正：`nn_methods` 舊的 run 註冊迴圈不認得 kind "dnet" → KeyError（第一次評估全部失敗，已改 setdefault 後重跑）。

| 集合 | method | 選髮絲 / 位移 | gs | ret_min | vm | slab_w | 移動根數 / p99 | 秒 / case |
|---|---|---|---|---|---|---|---|---|
| val | comb20_slab1 | FM / 梳理 | 0.049 | 0.78 | 0.0498 | 10.5 | 1470 / 13.3 mm | 39 |
| val | dopt_fm_mix10 | FM / 最佳化 | 0.046 | 0.78 | 0.0488 | 4.0 | 2397 / 9.4 mm | 41 |
| val | dnet_dn1_fm | FM / NN | 0.048 | 0.78 | 0.0485 | 4.9 | 5899 / 9.0 mm | 37 |
| val | dopt_k1_mix10 | NN / 最佳化 | 0.043 | 0.73 | 0.0428 | 3.7 | 2523 / 10.3 mm | 4.5 |
| val | **dnet_dn1_k1** | **NN / NN** | 0.045 | 0.73 | **0.0426** | 5.8 | 5917 / 10.1 mm | **1.9** |
| test | comb20_slab1 | FM / 梳理 | 0.091 | 0.70 | 0.0450 | 14.1 | 1575 / 14.4 mm | 42 |
| test | dopt_fm_mix10 | FM / 最佳化 | 0.091 | 0.70 | 0.0437 | 4.2 | 2828 / 10.3 mm | 42 |
| test | dnet_dn1_fm | FM / NN | 0.093 | 0.70 | 0.0457 | 6.6 | 6200 / 10.3 mm | 38 |
| test | dopt_k1_mix10 | NN / 最佳化 | 0.110 | 0.84 | 0.0458 | 22.5 | 3611 / 14.6 mm | 6.6 |
| test | **dnet_dn1_k1** | **NN / NN** | 0.110 | 0.84 | **0.0447** | 33.8 | 6708 / 14.3 mm | **1.9** |

- **攤銷成功**：NN 位移在 val 幾乎等於逐 case 最佳化（k1 keep：vm 0.0426 vs 0.0428，slab_w 5.8 vs 3.7），不需迭代。
- **全 NN 管線（NN 選髮絲 + NN 位移）約 1.9 s / case**（FM + 梳理約 40 s），val 上所有方法中 vm 最低。
- test（長髮）：FM keep 下 dn1 與梳理相當（vm 0.0457 vs 0.0450、slab_w 6.6 vs 14.1、gs 0.093 vs 0.091），略遜於逐 case 最佳化；
  全 NN 管線 vm 0.0447（優於梳理 0.0450）但 gs 0.110、slab_w 33.8 → 差距仍來自 k1 選髮絲。
- dn1 移動的髮絲比最佳化多（約 6k vs 2.5k 根，多為小幅移動），p99 相近；尚未渲染。

## 2026-11-24 [13] 全方法渲染比較（使用者要求）

- 比較圖：`$DATA/compare/all_val_case_{0000,0002,0007}.jpg`、`all_test_case_{0009,0010}.jpg`（`$DATA/montage2.py`；每列一方法並標該 case 指標）。
  列：orig_base / orig_donor（新增的「單一原始髮型」方法，`nn_methods._orig`）、mask_only、tsai_joint_fm（Baseline，本次才在這 5 case 上跑）、
  guard_slab1、comb20_slab1、dopt_fm_mix10、dnet_dn1_fm、nn_k1__nodisp、dopt_k1_mix10、dnet_dn1_k1。
- Baseline `tsai_joint_fm` 在長髮上改髮型：test 0010 donor 只剩 2 根（整顆 donor 消失，view_mix 0.0096 是因此而低）、0009 donor 3421 → 1991。
- FM 選髮絲的三種位移（梳理 / 直接最佳化 / NN 位移）在 0009 後視邊界都乾淨，彼此外觀相近；指標上直接最佳化與 NN 位移的 slab_w 最低（2.2 / 3.2 vs 梳理 7.5）。
- NN 選髮絲（k1）系列在長髮上的缺陷可見：0010 右視白髮絲穿過黑區、0009 後視黑髮絲壓在白區；位移（最佳化 / NN）能減少但不能消除 → 與 LOG [11]、[12] 一致，瓶頸在選髮絲。
- 未見位移方法造成豎起 / 成團 / 彎折。

## 2026-11-24 [14] SOTA 標記（使用者要求）

- 依 val 16 / test 12 case 的完整平均（LOG [11]、[12]）標記：
  - **SOTA（品質）`dopt_fm_mix10`**：test gs 0.091（= 梳理）、vm 0.0437、slab_w 4.2（皆最低，非退化方法中），不吃頭（ret_min 與 FM 相同）。
  - **SOTA（全 NN）`dnet_dn1_k1`**：1.9 s / case、val vm 0.0426（最低）；test vm 0.0447 但 gs 0.110（受 k1 選髮絲限制）。
- 對照表寫在 SUMMARY.md「重要結果」，checkpoints.md 的對應條目加上 [SOTA] 標記。Baseline 未更動。

## 2026-11-24 [15] 可攜性（交接到另一台機器）

- `run.sh`：工作區 / Python 自動偵測（見 `../prior/PLAN.md` 第 1 節）。
- `models/k1`、`models/dn1`（config.json、best.pt、train_log.jsonl）進 git；`nn_methods` 先找 `$DATA/runs/<name>`，否則 `models/<name>`。
  驗證：本機註冊結果不變（dnet_dn1_*、nn_k1__nodisp、dopt_k1_mix10 皆存在）。
- `split/split.json`、`split/lengths.json`：新機器複製到 $DATA 後以相同 seed 重建 cases。
