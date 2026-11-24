# CLAUDE.md — inference/fusion_refine/nn/

NN 版的融合精修（選髮絲 + 小幅位移）。概念見 `README.md`；進度只寫在 `LOG.md`（append only）、`SUMMARY.md`（「目前狀態」≤ 20 行，覆寫）、
`checkpoints.md`（版本清單：Promising / Experimental / Baseline / 已棄用；Baseline 只在使用者同意後改）。

## 執行

- 一律經 `run.sh <script.py> [args]`：它設定外接碟工作區 `hair_work`（資料集、`highlighting`、venv；repo 以 symlink 掛在那裡）、
  `$DATA`（資料與 checkpoint，預設 `hair_work/fusion_nn_data`，不進 git）與 PYTHONPATH（benchmark/、inference/）。
- `build_cases.py` = benchmark 的 `build_benchmark.py`（髮長改由 full_strands 算並快取；`--names_from split.json:<part>` 限定髮型）。
- `bench.py` = benchmark 的 `run_benchmark.py`，另外註冊本資料夾 `nn_methods.py` 的方法（opt-in，用 `--methods` 指名）。
  評估就用它，指標與 benchmark 相同；先用少量代表 case（`$DATA/val`、`$DATA/test`），再擴大。

## 慣例

- 髮型依 base style 切 train / val / test（`make_split.py` → `$DATA/split.json`），同一 base 的變體不跨 split。
- mask / 特徵 / 輸出都在 scalp-texture layout（row, col；col 左右對應頭的左右），與 benchmark 相同。
- benchmark、`fusion_refine/` 其他模組與 `img2hair_tsai_*.py` 不在這裡改；必須改時加預設關閉的 flag 並記到該處的 LOG。
- 特徵 / 模型 / 監督方式都會換：程式以版本（checkpoint 目錄 + 其 config.json）區分，不要把某一版寫成固定規格。
- GPU 與 RAM 與使用者共用：背景工作設 `--max_mem_gb`，不要在使用者跑全髮絲 benchmark 時開大型訓練。
