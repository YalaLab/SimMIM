export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

torchrun --standalone --nnodes=1 --nproc_per_node ${NUM_GPUS} main_simmim.py \
 --cfg configs/vit_base__800ep/mimic.yaml \
 --batch-size 224 \
 --output /home/ubuntu/scratch/code/SimMIM/logs/mimic_vitb_testrun \
 --tag mimic_vitb_full

# Optional: override data path via CLI if config differs
# --data-path /home/ubuntu/scratch/data/mimic/mimic/raw/image_files/mimic-cxr-jpg-2.1.0.physionet.org/files \
