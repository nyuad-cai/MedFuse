CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr --missing_token learnable \
--fusion_type early \
--save_dir checkpoints/pheno/early