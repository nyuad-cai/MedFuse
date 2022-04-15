CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py --dim 256 \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 50 --batch_size 16 --lr 5.326e-05 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--fusion_type mmtm --layer_after 4 \
--save_dir checkpoints/phenotyping/mmtm 