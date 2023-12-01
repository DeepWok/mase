./ch train nerf nerf-lego --max-epochs 1 --batch-size 16384 --learning-rate 0.00001

./ch test nerf nerf-lego --load ../mase_output/nerf_classification_nerf-lego_2023-11-29/software/training_ckpts/last-v2.ckpt --load-type pl