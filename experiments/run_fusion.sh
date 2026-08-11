input_config=./fusing_config.json
output_path=./output1/

python ../inference/img2hair_kung_mul2.py \
    --input_config ${input_config} \
    --output_path ${output_path}

bash rendering.sh "${output_path}" "${output_path}/render/"