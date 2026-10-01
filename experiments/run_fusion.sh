#!/bin/bash
set -e

# 兩種融合方法（上色策略相同：base 白、donor 黑），各自輸出到 ${output_path}/<method>/
KUNG_SCRIPT=../inference/img2hair_kung_mul2_tsai_color.py
TSAI_SCRIPT=../inference/img2hair_tsai_1.py

input_config=./fusing_config_new1.json
output_path=./outputs_tsai_ma/

mkdir -p "${output_path}"

# 要跑哪個方法: kung / tsai / both
method=tsai
# diffusion 的 seed（每張圖生成前都重設），兩個方法用同一個 seed 時，同一張圖會得到相同的髮型；負數 = 隨機
seed=5
# 是否保留3D模型檔案(.blend/.blend1/.npz)，設為0則算圖完後刪除，只留render圖節省空間
keep_3d_model=0
# tsai: 融合前是否先對齊髮線（diffusion -> 在 scalp texture 上位移特徵對齊髮線 -> decode），1=開啟 0=關閉
align_hairline=1

usage() {
    echo "Usage: $0 [--method kung|tsai|both] [--seed N] [--align-hairline] [--no-keep-model]"
    echo "  --method          要跑的融合方法 (預設: both)"
    echo "  --seed            diffusion seed (預設: 5)，負數為隨機"
    echo "  --align-hairline  (tsai) 融合前先對齊兩個髮型的髮線 (同 kung 的 align_hairline_by_shifting)"
    echo "  --no-keep-model   算圖完成後刪除3D模型檔案(.blend/.blend1/.npz)，只保留算圖結果"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            method="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --no-keep-model)
            keep_3d_model=0
            shift
            ;;
        --align-hairline)
            align_hairline=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知參數: $1"
            usage
            ;;
    esac
done

case "${method}" in
    kung) methods=(kung) ;;
    tsai) methods=(tsai) ;;
    both) methods=(kung tsai) ;;
    *)
        echo "未知的 method: ${method}"
        usage
        ;;
esac

# --- 1. 跑融合(fusion)流程 ---
for m in "${methods[@]}"; do
    if [[ "${m}" == "kung" ]]; then
        script=${KUNG_SCRIPT}
    else
        script=${TSAI_SCRIPT}
        # tsai 只支援 2 張圖 + 1 個 mask
        counts=$(python -c "import json; c = json.load(open('${input_config}')); print(len(c['input_file_paths']), len(c['mask_file_paths']))")
        if [[ "${counts}" != "2 1" ]]; then
            echo "警告: tsai 需要 2 張圖 + 1 個 mask，但 ${input_config} 是 (圖, mask) = (${counts})，跳過 tsai。"
            continue
        fi
    fi

    extra_args=()
    if [[ "${m}" == "tsai" && "${align_hairline}" == "1" ]]; then
        extra_args+=(--align_hairline)
    fi

    method_out="${output_path%/}/${m}"
    echo ">>> [${m}] ${script} -> ${method_out}"
    python ${script} \
        --input_config "${input_config}" \
        --output_path "${method_out}" \
        --seed "${seed}" \
        "${extra_args[@]}"

    bash rendering.sh "${method_out}" "${method_out}/render/"
done

# --- 2. (可選) 是否保留3D模型檔案 ---
if [[ "${keep_3d_model}" == "0" ]]; then
    echo ">>> 刪除3D模型檔案，僅保留算圖結果..."
    find "${output_path}" -type f \( -name "*.blend" -o -name "*.blend1" -o -name "*.npz" \) -delete
fi
