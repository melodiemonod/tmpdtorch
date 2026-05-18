#!/bin/sh
#PBS -N dps_missing
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=100gb:ngpus=1
#PBS -j oe

eval "$(/gpfs/home/mm3218/miniforge3/bin/conda shell.bash hook)"
conda activate DPS

REPO_DIR="/gpfs/home/mm3218/git/tmpdtorch"
data_config="/gpfs/home/mm3218/git/tmpdtorch/configs/ffhq_data_config.yaml"
model_config="/gpfs/home/mm3218/git/tmpdtorch/configs/ffhq_model_config.yaml"
diffusion_config="/gpfs/home/mm3218/git/tmpdtorch/configs/diffusion_config.yaml"
task_config="/gpfs/home/mm3218/git/tmpdtorch/configs/inpainting_config.yaml"

cd $REPO_DIR
for i in $(seq 1 53); do
    save_dir="/gpfs/home/mm3218/projects/2026/dps_sbc/ffhq/inpainting/run_${i}"

    python3 $REPO_DIR/sample_condition.py \
      --data_config=$data_config \
      --model_config=$model_config \
      --diffusion_config=$diffusion_config \
      --task_config=$task_config \
      --gpu=0 \
      --save_dir=$save_dir
done

