#!/bin/bash
#SBATCH --job-name=model_classifier_sweep
#SBATCH --partition=RTX3090,RTXA6000,RTXA6000-EI,batch,V100-32GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/console.%A_%a.out  # %A is JobID, %a is task ID
#SBATCH --error=output/console.%A_%a.err

# --- Job Array Configuration ---
# 10 models * 3 classifiers = 30 tasks total. Array indices are 0-29.
#SBATCH --array=0-29

# --- Static Parameters ---
# These parameters are the same for all jobs in the array
MODE="single"
DATASET="cifar10"
NUM_EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=2e-3
WEIGHT_DECAY=1e-3
SEED=0
ALPHA1=1.0
ALPHA2=0.0
ACTIVATION="silu"
DS_PERCENTAGE=0.50.1
GRID=5
DEGREE=3
NOISE=0.0

# --- Dynamic Parameters (mapped from array index) ---
# Define the parameters that change for each job
MODELS=(resnet18 resnet34 resnet50 vgg16 densenet121 densenet169 densenet201 densenet161 mobilenet_v2 googlenet)
CLASSIFIERS=(mlp kan gnu)

# --- Map SLURM_ARRAY_TASK_ID to Parameters ---
# This is the core logic that makes each array task unique
NUM_CLASSIFIERS=${#CLASSIFIERS[@]}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CLASSIFIERS))
CLASSIFIER_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CLASSIFIERS))

MODEL=${MODELS[$MODEL_INDEX]}
CLASSIFIER=${CLASSIFIERS[$CLASSIFIER_INDEX]}

# --- Print Job Info ---
# Good practice to log this information for debugging
echo "------------------------------------------------"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "---"
echo "Model: ${MODEL} (Index: ${MODEL_INDEX})"
echo "Classifier: ${CLASSIFIER} (Index: ${CLASSIFIER_INDEX})"
echo "------------------------------------------------"

# --- Execute the Job ---
# Note that resource allocation flags like --gpus, --mem, etc., are now in the #SBATCH
# directives above and are not needed in the srun command itself.
# The command `bash -c "./install.sh && python3 ..."` ensures install.sh runs first.
srun --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.02-py3.sqsh \
     --container-workdir=`pwd` \
     --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
     --export="ALL,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
     bash -c "./install.sh && \
     python3 main.py \
        --mode $MODE \
        --dataset $DATASET \
        --model $MODEL \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --weight_decay $WEIGHT_DECAY \
        --seed $SEED \
        --alpha1 $ALPHA1 \
        --alpha2 $ALPHA2 \
        --activation $ACTIVATION \
        --ds_percentage $DS_PERCENTAGE \
        --grid $GRID \
        --degree $DEGREE \
        --noise $NOISE \
        --learning_rate $LEARNING_RATE \
        --classifier $CLASSIFIER"

echo "Job ${SLURM_ARRAY_TASK_ID} finished with exit code $?"