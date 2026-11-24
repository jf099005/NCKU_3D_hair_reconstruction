#!/bin/bash
# fusion_refine/nn（與 ../prior）入口：設定工作區、$DATA、PYTHONPATH 後執行本資料夾的腳本。
#   bash run.sh <script.py> [args]          例：bash run.sh build_cases.py 40 --seed 3 --out $DATA/train_a
#   bash run.sh ../prior/elbo_check.py ...  （prior 的腳本也經這裡）
# 工作區 = 放 highlighting/ 與 DiffLocks_Dataset/ 的資料夾：
#   $HAIR_WORK（若設定）→ 外接碟 hair_work（存在時）→ repo 的上一層（標準配置 P76154862/）
# Python：$PYTHON → <工作區>/venv → conda env hair_exp
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/../../.." && pwd)"
if [ -z "$HAIR_WORK" ]; then
    if [ -d "/media/jf099005-u/新增磁碟區1/hair_work/highlighting" ]; then HAIR_WORK="/media/jf099005-u/新增磁碟區1/hair_work"
    else HAIR_WORK="$(cd "$REPO/.." && pwd)"; fi
fi
export HAIR_WORK
export DATA=${FUSION_NN_DATA:-"$HAIR_WORK/fusion_nn_data"}
export DATASET=${DATASET:-"$HAIR_WORK/DiffLocks_Dataset/difflocks_dataset"}
if [ -z "$PYTHON" ]; then
    if [ -x "$HAIR_WORK/venv/bin/python" ]; then PYTHON="$HAIR_WORK/venv/bin/python"
    else PYTHON="$HOME/anaconda3/envs/hair_exp/bin/python"; fi
fi
# highlighting/ 在工作區下；benchmark/ 與 inference/ 供 import
export PYTHONPATH="$HAIR_WORK:$REPO/benchmark:$REPO/inference:$REPO:$DIR:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
script=$1; shift
exec "$PYTHON" "$DIR/$script" "$@"
