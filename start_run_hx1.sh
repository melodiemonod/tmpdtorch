#!/bin/bash

N=100  # number of jobs you want to launch

METHOD="TMPD"
DATANAME="ffhq"
OPERTORNAME="inpainting"
BASE_JOBID="${METHOD}/${DATANAME}/${OPERTORNAME}"

REPO_DIR="/gpfs/home/mm3218/git/tmpdtorch"
SAVE_DIR="/gpfs/home/mm3218/projects/2026/dps_sbc/${BASE_JOBID}"

data_config=$REPO_DIR/configs/${DATANAME}_data_config.yaml
model_config=$REPO_DIR/configs/${DATANAME}_model_config.yaml
diffusion_config=$REPO_DIR/configs/diffusion_config.yaml
task_config=$REPO_DIR/configs/${OPERTORNAME}_config.yaml

mkdir -p $SAVE_DIR

TEMPLATE_FILE="run_template.pbs"

for i in $(seq 1 $N); do
    
    job_dir=${SAVE_DIR}/run_${i}
    job_script="${job_dir}/run_${i}.pbs"
    mkdir -p $job_dir

    cat > $job_script <<EOF
#!/bin/sh
#PBS -N tmpd_${i}
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=100gb:ngpus=1
#PBS -j oe

eval "\$(/gpfs/home/mm3218/miniforge3/bin/conda shell.bash hook)"
conda activate tmpdtorch

REPO_DIR="$REPO_DIR"
data_config="$data_config"
model_config="$model_config"
diffusion_config="$diffusion_config"
task_config="$task_config"

save_dir="$job_dir"

cd \$REPO_DIR
python3 \$REPO_DIR/sample_condition.py \\
  --data_config=\$data_config \\
  --model_config=\$model_config \\
  --diffusion_config=\$diffusion_config \\
  --task_config=\$task_config \\
  --gpu=0 \\
  --save_dir=\$save_dir
EOF

    chmod +x $job_script
    
done


for i in $(seq 1 $((N / 50))); do
    start_index=$(( (i - 1) * 50 + 1 ))
    end_index=$(( i * 50 ))

    job_script="${SAVE_DIR}/job_run_${start_index}-${end_index}.pbs"

    cat > $job_script <<EOF
#!/bin/sh
#PBS -N tmpd_${start_index}-${end_index}
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=100gb:ngpus=1
#PBS -J ${start_index}-${end_index}
#PBS -j oe

eval "\$(/gpfs/home/mm3218/miniforge3/bin/conda shell.bash hook)"
conda activate tmpdtorch

REPO_DIR="$REPO_DIR"
data_config="$data_config"
model_config="$model_config"
diffusion_config="$diffusion_config"
task_config="$task_config"

save_dir=${SAVE_DIR}/run_\${PBS_ARRAY_INDEX}

cd \$REPO_DIR
python3 \$REPO_DIR/sample_condition.py \\
  --data_config=\$data_config \\
  --model_config=\$model_config \\
  --diffusion_config=\$diffusion_config \\
  --task_config=\$task_config \\
  --gpu=0 \\
  --save_dir=\$save_dir
EOF

    chmod +x $job_script
    job_name=$(basename "$job_script")
    if [ "$i" -eq 1 ]; then
        echo "cd $SAVE_DIR"
        echo "qsub $job_name"
    else
        echo "qsub -W depend=afterany:12345 $job_name"
    fi
    
done


