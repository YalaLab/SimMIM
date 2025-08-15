#!/bin/bash

# Configuration
REPO_ID="YalaLab/simmim-scratch-000"
EXTRA_FLAGS="--private"
# PATH_BASE="/home/ubuntu/scratch/code/SimMIM/logs/mimic_vitb_full/simmim_pretrain/mimic_vitb_full_8gpu_pt_bugfix"
# PATH_BASE="/home/ubuntu/scratch/code/SimMIM/logs/mimic_vitb_full/simmim_pretrain/mimic_vitb_full_8gpu_pt_bugfix_lower_lr"
PATH_BASE="/home/ubuntu/scratch/code/SimMIM/logs/mimic_vitb_full/simmim_pretrain/mimic_vitb_full_4gpu"


# Function to process and push a checkpoint
push_checkpoint() {
    local iter=$1
    local ckpt_path="${PATH_BASE}/ckpt_epoch_${iter}.pth"
    
    # Push to HuggingFace
    if [ -f "$ckpt_path" ]; then
        echo "Processing checkpoint for iteration ${iter}..."
        python "$(dirname "$0")/push_to_hf_lite.py" "$ckpt_path" "$REPO_ID" $EXTRA_FLAGS
    else
        echo "Warning: Checkpoint file not found for iteration ${iter}"
    fi
}

# List of iterations to process
iterations=(150)

# Process each iteration
for iter in "${iterations[@]}"; do
    push_checkpoint $iter
done