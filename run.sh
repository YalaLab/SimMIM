export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch --nproc_per_node gpu main_simmim.py \
--cfg configs/swin_base__800ep/simmim_pretrain__swin_base__img192_window6__800ep.yaml \
--batch-size 128 \
--data-path <imagenet-path>/train \
--output <output-directory> \