# Lab Logbook Answers

## Lab 0

### Tutorial 1

> Task: Delete the call to `replace_all_use_with` to veriyf that FX will report a RuntimeError

Shows error as expected. The runtime error describes:

```
RuntimeError: Tried to erase Node bert_embeddings_dropout but it still had 6 users in the graph:
{getattr_2, size_4, bert_encoder_layer_0_attention_self_query,
 bert_encoder_layer_0_attention_self_key, bert_encoder_layer_0_attention_self_value, add_7}
```

This is because the 6 nodes which were found in the earlier analysis pass still depend on the drouput output, i.e. the nodes have the drouput node in their `args`. FX prevents deletion because it would leave them with invalid inputs. Uncommenting the `replace_all_use_with` and running the analysis pass again shows proper removal, where dropout count is now 0.

### Tutorial 2:

> Task: Remove the `attention_mask` and `labels` arguments from the `hf_input_names` list and re-run the following cell. Use `mg.draw()` to visualize the graph in each case. Can you see any changes in the graph topology? Can you explain why this happens?

- Observation: After commenting out `attention_mask` and `labels` arguments, the nodes `get_attr`, `ones` and `add_3` appear at the top of the graph, but the `attention_mask` and `CrossEntropyLoss` gets removed.
- Explanation: Without `attention_mask`, the model needs to create a default attention mask, which it does using `ones` with the same shape as `input_ids`. Without `labels`, the `CrossEntropyLoss` module is removed from the graph, and the associated `view` operations for reshaping tensors for loss computation are also removed.

## Lab 1

### Tutorial 3: QAT

QAT core idea: uses fake quantization - simulate lower precision numbers being run in original storage precision (e.g. uses FP32 format but constrains values to INT8 precision during training)

> Task: In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32. Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3. Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.


### Tutorial 4: Pruning

> Task: Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9. Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4. Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.

## Lab 2

### Tutorial 5: NAS Optuna

> Task: Tutorial 5 shows how to use random search to find the optimal configuration of hyperparameters and layer choices for the Bert model. Now, explore using the GridSampler and TPESampler in Optuna. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.


> Task: In Tutorial 5, NAS is used to find an optimal configuration of hyperparameters, then we use the CompressionPipeline in Mase to quantize and prune the model after search is finished. However, the final compressed model may not be optimal, since different model architectures may have different sensitivities to quantization and pruning. Ideally, we want to run a compression-aware search flow, where the quantization and pruning is considered in each trial. In the objective function, after the model is constructed and trained for some iterations, call the CompressionPipeline to quantize and prune the model, then continue training for a few more epochs. Use the sampler that yielded the best results in Task 1 to run the compression-aware search. The objective function should return the final accuracy of the model after compression. Consider also the case where final training is performed after quantization/pruning. Plot a new figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. There should be three curves: 1. the best performance from Task 1 (without compression), compression-aware search without post-compression training, and compression-aware search with post-compression training.



## Lab 3

### Tutorial 6: Mixed Precision Search

> Task: In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization. Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.



> Task: In Section 1 of Tutorial 6, when defining the search space, a number of layers are imported, however only LinearInteger and the full precision nn.Linear are selected. Now, extend the search to consider all supported precisions for the Linear layer in Mase, including Minifloat, BlockFP, BlockLog, Binary, etc. This may also require changing the model constructor so the required arguments are passed when instantiating each layer. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each precision to compare their performance.




### [Hardware Stream] Tutorial ?

