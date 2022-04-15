CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr \
--fusion_type uni_ehr --task in-hospital-mortality \
--labels_set mortality \
--save_dir checkpoints/mortality/uni_ehr_all 
