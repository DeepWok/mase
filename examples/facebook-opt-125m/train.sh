# Run test to get the perplexity of pretrained facebook/opt-125m on test set
./chop --test-sw --model facebook/opt-125m --task lm --dataset ptb --pretrained --load-type hf --project opt-125m_lm_ptb --cpu 8 --gpu 2 --batch-size 4
