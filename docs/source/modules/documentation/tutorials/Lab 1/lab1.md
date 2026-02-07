In this lab, you will learn how to use Mase to compress a Bert model using quantization and pruning. You will build off the checkpoint from Lab 2, where we fine tuned a Bert model for sequence classification using the LoRA adapter. You will quantize the model to fixed-point precision and then prune the model to reduce the number of parameters. After each stage, you’ll run further fine tuning to recover the performance lost during compression.

Learning tasks
Go through “Tutorial 3: Running Quantization-Aware Training (QAT) on Bert” to learn how to quantize the Bert model and run post-quantization finetuning.

Go through “Tutorial 4: Unstructured Pruning on Bert” to understand how to prune a quantized model for further compression.

Implementation tasks
In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.

Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.

Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.

Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.

Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.