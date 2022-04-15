CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 50 --batch_size 16 --lr 0.00053985 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr \
--fusion_type uni_ehr \
--save_dir checkpoints/pheno/uni_ehr_full 