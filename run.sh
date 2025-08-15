export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

torchrun --standalone --nnodes=1 --nproc_per_node ${NUM_GPUS} main_simmim.py \
 --cfg configs/vit_base__800ep/mimic_pretrained.yaml \
 --batch-size 240 \
 --output /home/ubuntu/scratch/code/SimMIM/logs/mimic_vitb_full \
 --tag mimic_vitb_full_8gpu_pt_bugfix_lower_lr

# Optional: override data path via CLI if config differs
# --data-path /home/ubuntu/scratch/data/mimic/mimic/raw/image_files/mimic-cxr-jpg-2.1.0.physionet.org/files \
