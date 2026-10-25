#!/bin/bash
# 融合 benchmark 入口：建立測資 / 執行。需要 numba + mediapipe + NATTEN 的環境 (預設 hair_exp)
#   bash benchmark/bench.sh build <n> [build_benchmark.py 的其他參數]
#   bash benchmark/bench.sh run <benchmark_dir> [run_benchmark.py 的其他參數]
#   bash benchmark/bench.sh render <run_dir> --cases <ids> [...]              (事後補算指定 case 的圖，不重跑融合)
#   bash benchmark/bench.sh summarize <run_dir>                              (summary.* + report.html，並算圖各長度組合的最高/最低 grid score case)
set -e
PYTHON=${PYTHON:-$HOME/anaconda3/envs/hair_exp/bin/python}
export PYTHONUNBUFFERED=1   # 輸出導向檔案 (nohup / >) 時也即時寫入，被 OOM kill 時 log 不會遺失
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd=$1; shift || true
case "${cmd}" in
    build)     exec "${PYTHON}" "${DIR}/build_benchmark.py" "$@" ;;
    run)       exec "${PYTHON}" "${DIR}/run_benchmark.py" "$@" ;;
    render)    exec "${PYTHON}" "${DIR}/render_results.py" "$@" ;;
    summarize) exec "${PYTHON}" "${DIR}/summarize.py" "$@" ;;
    *) echo "Usage: $0 build <n> [...] | run <benchmark_dir> [...] | render <run_dir> [...] | summarize <run_dir>"; exit 1 ;;
esac
