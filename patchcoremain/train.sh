#!/bin/bash

# 1. 데이터 경로 설정 (MVTec LOCO)
datapath="./mvtec_loco_anomaly_detection"

# 2. MVTec LOCO 데이터셋 목록
datasets=('breakfast_box' 'juice_bottle' 'screw_bag' 'pushpins' 'splicing_connectors')

# 3. 데이터셋 플래그 생성
# (이 플래그는 파이썬 코드의 -d, --subdatasets 옵션의 기본값을 덮어씁니다)
dataset_flags=()
for dataset in "${datasets[@]}"; do
  dataset_flags+=("-d" "$dataset")
done


# 4. 스크립트 실행
# ===================================================================
# 우리가 수정한 코드는 patch_core와 sampler에 대한 인자가 필요 없습니다.
# (모두 default로 지정되어 있음)
# dataset 명령어에만 옵션과 필수 인자를 전달합니다.
# ===================================================================

python run_patchcore.py \
    --gpu 0 \
    --seed 0 \
    --save_patchcore_model \
    --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
    --log_project MVTecAD_Results \
    results \
    patch_core \
    sampler \
    dataset \
        --resize 256 \
        --imagesize 224 \
        "${dataset_flags[@]}" \
        mvtec \
        $datapath