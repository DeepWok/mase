## Tutorial 1 (Lab 0): Introduction

### Run-Time Error
Here is the run-time error encountered during the session:

```plaintext
RuntimeError: Tried to erase Node bert_embeddings_dropout but it still had 6 users in the graph: 
{getattr_2: None, size_4: None, bert_encoder_layer_0_attention_self_query: None, 
bert_encoder_layer_0_attention_self_key: None, bert_encoder_layer_0_attention_self_value: None, 
add_7: None}!
```

## Tutorial 2 (Lab 0): Lora Finetune

### Removing attention_mask, labels from hf_input_names and its effect on the graphs:
The graph was created with and without the extra information and compared.

Firstly, having no labels meant that at the end of the process, there is no cross-entropy calcualted and viewed (so 4 blocks are removed). This is becuase the ground truth labels are required for loss calcualtions and not having them therefore means no losses can be calculated. Secondly, when there is no attention_mask specified, the model calls an extra block called getattr_1 after the input, instead of having a seperate input attention_mask block. When no mask is specified, more information from the model is used as an input to the masking process, implying that the mask is created based on the input information, whilst the external mask would be used for manually choosing which information to mask or not.


## Tutorial 3 (Lab 1): QAT
Insert plots for tutorial 3 (lab 1).
![Fixed point width vs highest achieved accuracy](fixed_point_width_vs_accuracy.png) 
![PTQ and QAT comparison vs highest achieved accuracy](ptq_vs_qat_accuracy.png) 

The best model that was most condensed was fixed width 16 (8,8) which was saved.


## Tutorial 4 (Lab 1): Pruning
![Sparsity vs highest achieved accuracy](highest_accuracy_by_sparsity.png) 
![Random vs L1-Norm comparison](pruning_accuracy_by_sparsity.png) 

L1-norm performed better than random in all cases and allowed for more drastic pruning.


## Tutorial 5 (Lab 2): Nas Optuna
![Random vs Grid vs TPE search method comparison](combined_optuna_results.png) 

TPE found the highest accuracy combination the fastest and reached the highest accuracy, therefore was the best search method.
TPE was then used in part b, and compression-aware search ran and tested.

![Effects of compression and post-compression fine-tuning](compression_aware_results.png) 

No compression eventually performed the best, mainly due to the compression being quite severe, but the compression aware training method reach similar accuracy levels to the non-compressed model with a much smaller model size.

## Tutorial 6 (Lab 3): Mixed Precision Search
![Number of trails vs maximum achieved accuracy](optuna_search_results.png)

TPE Sampler was used to search for the optimal configuration, which was found on the 3rd iteration.
The search was then extended to contain the following quantised configuartions for each layer:

- torch.nn.Linear (no quantisation)
- LinearInteger, 
- LinearMinifloatDenorm
- LinearMinifloatIEEE
- LinearLog
- LinearBlockFP
- LinearBlockLog
- LinearBinary
- LinearBinaryScaling

(I wasn't able to use LinearBlockMinifloat without errors and LinearBinaryResidualSign had not been implemented yet so these were ommited.) I initially used the Optuna sampler (TPE in this case) to search for the optimal layer types which yielded the following results.

![Number of trails vs maximum achieved accuracy](extended_optuna_search_results_2_curves.png)

After realising seperate curves for each precision were wanted, the code was rewritten to do 5 iterations of 3 epochs for each layer type. Here are these results.

![Maximum achieved accuracy after 5 iterations for each precision layer type](optuna_combined_precision_progress.png)
The LinearLog and LinearBlockLog were both found to be completely inneffective (maybe implemented incorrectly), with LinearBinary and LinearBinaryScaling found not much higher in accuracy achieved.

To see the trends in the rest of the results, here is a zoomed in view of these.
![Best performing precision layer types](optuna_combined_precision_progress_zoomed.png)

## Lab 4: ADLS Software Stream




1.
   a. Modify the code and investigate why this is the case?
   b. If you change the `device` to `cuda`, do you observe the same thing?
2. In the second part of Lab 4 (kernel fusion), we looked at a fused SDPA kernel.
   a. Now, extend the profiling to the SDPA kernel, compare its runtime behavior with the naive implementation.
   b. If you change the `device` to `cuda`, do you observe the same thing?
3. In the third part of lab4 (Custom kernel), we go through how to write MXINT8 dequantization kernel and bind it to Python.
   a. How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?
   b. What is the purpose of the variable `dont_need_abs` and `bias` in the C++ for loop?
   c. How does `cta_tiler` partition data for copying to shared memory in CUDA kernel? How does `layout_sX` partition threads in a threadlock for computation? (Challenge)
   d. Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7% of the FP32 model?




Check all questions are fully answered/described.
