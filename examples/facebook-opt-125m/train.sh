# train opt-125m on BoolQ
./chop --train --model facebook/opt-125m --task cls --dataset boolq --max-epochs 3 --batch-size 4 --cpu 8 --pretrained --load-type hf --project facebook-opt-125m

# test saved opt-125m
./chop --test-sw --model facebook/opt-125m --task cls --dataset boolq --batch-size 4 --cpu 8 --load-type pl --load ../mase_output/facebook-opt-125m/software/checkpoints/best.ckpt --project facebook-opt-125m
