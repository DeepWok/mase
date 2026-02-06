
# Lab 3: #

## Task 1: Mixed Integer Widths Search Analysis ##

### 1. Experiment Overview ###

- For this task, we conducted a 100 trials (1 epoch training) mixed-precision NAS search on the BERT model for IMDB classification using the Optuna's TPE sampler, which exhibited the best performance in Lab 2 (NAS Search). 

- Each of the 23 linear layers could independently be assigned full precision (FP32) or integer quantization with configurable bit-widths `[8, 16, 32]` and fractional widths `[2, 4, 8]`.

- We initialised the search using the best model from Lab 2 (`tutorial_5_best_model.pkl`) achieved by the `TPE Sampler`.

#### Summary Results ####

- __Best accuracy:__ 88.00% achieved at Trial 83
- __Optimal configuration:__ 8 of 23 layers quantized (34.8%)
- __Failing Trials:__ 5 trials collapsed to 50% accuracy (random guessing for IMDB binary classification)


### 2. Implementation Approach: ###

Rather than working directly in the provided Jupyter notebook, we converted the code to a standalone Python script to enable job submission on Imperial's HPC cluster. This allowed us to run longer experiments (100 trials) without worrying about notebook timeouts or disconnections.

The key modifications from the original tutorial notebook were:

1. __Per-layer width/frac_width selection:__ The tutorial used fixed quantization parameters `(width=8, frac_width=4)` for all `LinearInteger` layers. We extended `construct_model` to expose these as per-layer Optuna hyperparameters as per the task requirements:

```python
w = trial.suggest_categorical(f"{name}_width", [8, 16, 32])
fw = trial.suggest_categorical(f"{name}_frac_width", [2, 4, 8])
```

2. __Proper Weight and Bias Copying:__ The notebook only copied weights `(new_layer.weight.data = layer.weight.data).` We added bias handling and used `.copy_()` for safety:

```python
new_layer.weight.data.copy_(layer.weight.data)
   if layer.bias is not None:
       new_layer.bias.data.copy_(layer.bias.data)
```

3. __Logging and Checkpointing:__ The notebook has no callback implemented. If the session is interrupted we lose all our progress. Therefore, we implemented `LoggingCallback` that saves the best model and intermediate results after every trial.

4. __Adding Trial Metadata:__ The notebook only saves the model. We have added more attributes for our analysis:

```python
trial.set_user_attr("model", model)
trial.set_user_attr("num_params", num_params)
trial.set_user_attr("num_quantized", num_quantized)
trial.set_user_attr("width_counts", width_counts)
trial.set_user_attr("frac_width_counts", frac_width_counts)
```

### 3. Accuracy Progression Analysis ###

![Plot](./lab_3/lab3_results/plots/task1/mixed_linear_precision_(integer_widths)_summary.png)
Figure 1: Accuracy per trial & cumulative max accuracy for mixed integer widths (1 training epoch per trial)

As illustrated in Figure 1, the search process identified a high-performing configuration almost immediately. Trial 1 achieved an initial accuracy of 87.52%, which quickly improved to 87.99% by Trial 3. Following this rapid early gain, the cumulative best performance entered a significant plateau, remaining stagnant for 80 iterations until Trial 83 yielded a marginal increase to 88.00%.

While this extended plateau was initially unexpected, it can likely be attributed to two key factors:

- __Model Capacity Constraints:__ Given that BERT-tiny is a highly compressed architecture, there is probably an inherent performance ceiling. It is probable that the model has reached its maximum representative capacity for the IMDB dataset, regardless of the quantization strategy applied.

- __Search Space:__ the search space appears to be characterized by a high density of near-optimal solutions rather than a sparse global maximum. This indicates that the objective function is relatively "flat", where diverse quantization configurations yield statistically insignificant differences in model performance.

#### 50% Accuracy Drops: ####
A notable phenomenon in Figure 1 is the presence of five specific iterations (Trials 6, 15, 27, 38, and 72) where performance regressed to exactly 50% accuracy. In the context of the binary IMDB sentiment task, this represents random guessing.

This behavior is likely a result of over-aggressive quantization on high-sensitivity layers. An analysis of the optimal configuration supports this hypothesis: the `attention.output.dense` layers and the `classification head` were consistently maintained in `full precision (FP32)`. This suggests these components are critical for maintaining numerical stability and information flow of the network's forward pass.

We can also note the temporal distribution of these failing trials. The frequency of "crashed" trials dropped from 10% in the first 40 iterations to just 1.6% in the final 60. This confirms that the TPE effectively modeled the high-loss regions of the search space and learned to prioritize more robust configuration candidates as the optimization progressed.


### 4. Discussion: TPE Dynamics and Limitations ###

- __TPE Dynamics:__ The search behavior reflects the TPE sampler's transition from exploration to exploitation. The initial 10 startup trials (random baseline `n_startup_trials`=10 [2]) established a probability density, after which the sampler successfully identified and clustered around the high-performance manifold (87.7%–87.9%).

- __Limitations:__ The rapid performance plateau suggests that TPE's underlying independence assumption might be a contributing bottleneck. [1] explicitly categorize TPE as an independent sampling method, noting that such algorithms are "known to perform well even without using the parameter correlations." However, in the context of deep quantization, this independence may hinder the discovery of complex, inter-layer dependencies (e.g., a quantized Key layer requiring a high-precision Query layer). While TPE is effective generally, it is probable that its inability to explicitly model these joint relationships limits efficiency in this specific high-dimensional space. Additionally, the `1-epoch` training budget likely constrained the model's ability to adapt to aggressive quantization, further limiting the viability of lower-precision configurations.

### 5. Best Configuration: ###

The optimal configuration found by the search (Trial 83) reveals a highly selective quantization strategy. The optimal model includes 8 quantized layers out of the 23 layers (34.8%).

#### Quantization Distribution ####

The distribution of bit-widths among the quantized layers shows a strong preference for lower precision, with 62.5% of the quantized layers utilizing 8-bit widths. This confirms that significant compression is possible in specific parts of the network.


| Width (bits) | Count | Percentage |
|--------------|-------|------------|
| 8            | 5     | 62.5%      |
| 16           | 2     | 25.0%      |
| 32           | 1     | 12.5%      |


#### Layer-Wise Configuration Detail ####

We have saved the best model in `.pt` format so that we can extract per-layer information and perform our analysis. The table below details exactly which layers were quantized. A key observation is the asymmetry in the attention mechanism: Query and Value projections were quantized more frequently than Key projections, suggesting the model is more sensitive to precision loss in the Key vectors.

| Layer Name                                   | Type          | Width | Frac Width |
|----------------------------------------------|---------------|------:|-----------:|
| bert.encoder.layer.0.intermediate.dense       | LinearInteger |     8 |          8 |
| bert.encoder.layer.0.output.dense             | LinearInteger |    16 |          4 |
| bert.encoder.layer.1.attention.self.key       | LinearInteger |     8 |          4 |
| bert.encoder.layer.1.attention.self.value     | LinearInteger |     8 |          2 |
| bert.encoder.layer.2.attention.self.query     | LinearInteger |     8 |          2 |
| bert.encoder.layer.2.output.dense             | LinearInteger |    32 |          8 |
| bert.encoder.layer.3.attention.self.query     | LinearInteger |    16 |          4 |
| bert.encoder.layer.3.attention.self.value     | LinearInteger |     8 |          2 |


#### Critical Full Precision Layers ####

A crucial finding is the set of layers the search algorithm "chose" to keep in Full Precision (`nn.Linear`). The Attention Output layers and the Classifier head were universally protected. This aligns with the hypothesis that these layers act as information bottlenecks where higher precision is necessary to aggregate features correctly.

| Protected Layer Group   | Count | Specific Layers                                      |
|--------------------------|------:|------------------------------------------------------|
| Attention Output         |     4 | layer.0, layer.1, layer.2, layer.3 attention outputs |
| Classifier Head          |     1 | classifier                                           |
| Intermediate Dense       |     2 | layer.1, layer.2                                     |
| Output Dense             |     2 | layer.1, layer.3                                     |
| Attention Projections    |     6 | layer.0 (Q, V), layer.1 (Q), layer.2 (K, V)          |


## Task 2: Mixed Precision Analysis ##

### 1. Experiment Overview ###

For this task, we extended the search space to include all quantization formats supported by MASE's linear layer implementations:

`nn.Linear`
`LinearInteger`
`LinearMinifloat`
`LinearMinifloatDenorm`
`LinearLog`
`LinearBlockFP`
`LinearBlockMinifloat`
`LinearBlockLog`
`LinearBinary`

The search space also expanded to include precision-specific parameters: `exponent_width [3, 4, 5]`, `exponent_bias [3, 7, 15]`, and `block_size [8, 16, 32]` for formats that require them. We explored the quantizers implementation in `/src/chop/nn/quantizers/` to identify the precision-specific parameters required for each precision type. We note that all these parameters now even dramatically increases the configuration space compared to Task 1. 

```python
QUANTIZED_LAYER_CLASSES = [
    LinearInteger,
    LinearMinifloatIEEE,
    LinearMinifloatDenorm,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
]

# Full search space: full precision + all quantized types
LINEAR_LAYER_CHOICES = [nn.Linear] + QUANTIZED_LAYER_CLASSES

# Search space choices for different precision types
WIDTH_CHOICES = [8, 16, 32]
FRAC_WIDTH_CHOICES = [2, 4, 8]
EXPONENT_WIDTH_CHOICES = [3, 4, 5]
EXPONENT_BIAS_CHOICES = [3, 7, 15]
BLOCK_SIZE_CHOICES = [8, 16, 32]

```

#### Summary Results ####

- Best accuracy: 88.02% at Trial 53
- Best configuration: 18 of 23 layers quantized (78.3%%)
- Multiple failure modes observed (50%, 63%, 75% accuracy drops)

### 2. Implementation Approach & Source Code Modifications: ###

To handle the complexity of the expanded search space, we refactored the `construct_model` function to use a `Dictionary of Config Builders`. Unlike Task 1, where every layer shared the same parameter structure (`width` and `frac_width`), Task 2 required distinct logic for each precision type (e.g., `LinearBlockFP` requires `block_size` and `exponent_width`, while `LinearBinary` requires `stochastic` and `bipolar` flags).

Example on Config Builders:

```python
def build_log_config(trial: optuna.Trial, name: str) -> dict:
    """
    Build config for LinearLog.
    Required keys: data_in_width, data_in_exponent_bias,
                   weight_width, weight_exponent_bias,
                   bias_width, bias_exponent_bias
    """
    w = trial.suggest_categorical(f"{name}_width", WIDTH_CHOICES)
    eb = trial.suggest_categorical(f"{name}_exponent_bias", EXPONENT_BIAS_CHOICES)

    return {
        "data_in_width": w,
        "data_in_exponent_bias": eb,
        "weight_width": w,
        "weight_exponent_bias": eb,
        "bias_width": w,
        "bias_exponent_bias": eb,
    }
```

#### Source Code Debugging & Fixes ####

Upon initially running the search, we encountered critical failures rooted in the `mase` library's source code. We identified and patched three specific bugs to enable the search to run:

1. __Dimension Mismatch in Binary Quantizers:__ The initial runs crashed with an `IndexError: tuple index out of range` in `LinearBinary` layers.

__Cause:__ The library implementation of `BinaryZeroScaled` and `BinaryBipolarScaled` hardcoded an assumption that inputs would be `4-dimensional` (Batch, Channel, Height, Width), typical for CNNs. BERT linear layers produce `2-dimensional tensors` (Batch*Seq, Hidden), causing the code to crash when accessing dimension indices 2 and 3.

__Fix:__ We patched `src/chop/nn/quantizers/utils.py` to dynamically detect the input dimension and adjust the mean calculation (alpha) and broadcasting logic accordingly.

```Python
# Patch applied to src/chop/nn/quantizers/utils.py
@staticmethod
def alpha(input):
    # Dynamic dimension handling for Linear (2D) and Conv (4D) layers
    absvalue = input.abs()
    if input.dim() == 4:
        alpha = absvalue.mean(dim=(1, 2, 3), keepdims=True)
    elif input.dim() == 2:
        alpha = absvalue.mean(dim=(1,), keepdims=True)  # Added support for 2D
```

2. __Broken Backward Pass in Block Minifloat:__ After fixing the binary layers, the search crashed with a `TypeError` in `LinearBlockMinifloat` during backpropagation.

__Cause:__ The `backward` method in `BlockMinifloatQuantize` was defined to expect configuration arguments (e.g., `width`, `exponent_width`) that were passed to `forward`. However, PyTorch's autograd engine does not pass these arguments to `backward` automatically; it only passes the context (ctx) and the gradient. We confirmed this is the case by looking into the custom `backward` functions in `block_fp`, `block_log`, etc. in `src/chop/nn/quantizers/`

__Fix:__ We modified `src/chop/nn/quantizers/block_minifloat.py` to remove the extra arguments from the `backward` signature.

```python
# Patch applied to src/chop/nn/quantizers/block_minifloat.py
@staticmethod
def backward(ctx, grad_output): # Removed: width, exponent_width, etc.
    # Return gradient for input 'x', and None for the 5 configuration args
    return grad_output, None, None, None, None, None

```
### 3. Accuracy Progression Analysis ###

![Plot](./lab_3/lab3_results/plots/task2/multi_precision_search_(mixed_layer_types)_summary.png)
Figure 2: Accuracy per trial & cumulative max accuracy for mixed precision search (1 training epoch per trial)

The progression shows a more turbulent search compared to Task 1. Trial 1 immediately failed (50.03%), likely due to aggressive mixed-precision assignment across all the new formats. The search recovered quickly, reaching 85.54% by Trial 2 and steady improvements through Trials 4 (87.53%), 6 (87.71%), 13 (87.75%), 14 (87.81%), 17 (87.84%), 42 (87.91%), until the best of 88.02% at Trial 53.

Compared to Task 1, we observe more variance in accuracy. Beyond the 50% catastrophic failures (Trials 1, 36, 77, 90), there were partial degradations: Trial 40 dropped to 75.72%, Trial 57 to 79.43%, and Trial 70 to 63.29%. This suggests some precision combinations cause partial rather than complete model collaps

### 4. Comparison with Task 1 ###

| Metric                         | Task 1 (Integer Only) | Task 2 (Multi-Precision) |
|--------------------------------|----------------------|--------------------------|
| Best Accuracy                 | 88.00%               | 88.02%                   |
| Best Trial                    | 83                   | 53                       |
| Layers Quantized              | 8/23 (34.8%)         | 18/23 (78.3%)            |
| Full Precision Layers         | 15                   | 5                        |
| Catastrophic Failures (50%)   | 5                    | 4                        |
| Partial Failures (< 80%)      | 0                    | 3                        |
| Runtime                       | ~2.5 hours           | ~11 hours                |


The multi-precision search found a marginally better configuration (+0.02%) and did so faster (Trial 53 vs Trial 83). Interestingly, while Task 1's best model was conservative (65% full precision), Task 2's best model was aggressive—quantizing 78% of layers using a diverse mix of formats.

### 5. Best and Worst Model Configurations ###

#### Best Configurations ####

The winning configuration used 7 different precision types:

| Precision Type        | Layers | Percentage |
|----------------------|--------|------------|
| Linear (FP32)        | 5      | 21.7%      |
| LinearMinifloatIEEE  | 4      | 17.4%      |
| LinearBlockFP        | 4      | 17.4%      |
| LinearMinifloatDenorm| 3      | 13.0%      |
| LinearBlockLog       | 3      | 13.0%      |
| LinearInteger        | 3      | 13.0%      |
| LinearLog            | 1      | 4.3%       |
| **Total**            | **23** | **100%**   |


![Plot](./lab_3/lab3_results/plots/task2/best_precision_distribution.png)


This is notably different from Task 1's best model, which used only LinearInteger for quantized layers. The diversity of formats suggests that different layers may indeed benefit from different quantization schemes—minifloat formats preserve dynamic range better for some layers, while block formats share exponents efficiently for others.

#### Failure Configurations ####

Examining the failed trials reveals patterns:

__Catastrophic failures (~50% accuracy):__

- Trial 36: Heavy `LinearLog` (5) and `LinearBinary` (3) usage, only 2 full precision layers
- Trial 77, 90: Similar patterns with aggressive quantization

__Partial failures:__

- Trial 40 (75.72%): 3 `LinearBinary`, 4 `LinearBlockLog`, only 2 full precision
- Trial 70 (63.29%): 6 `LinearLog`, 2 `LinearBinary`, only 3 full precision

The common thread appears to be heavy use of LinearLog and LinearBinary combined with minimal full precision layers. These aggressive formats seem to require more careful placement or longer training to work effectively.


### 6. Key Observations: ###

The multi-precision search demonstrates that combining different quantization formats can match or slightly exceed integer-only quantization, while being more aggressive with the number of quantized layers (78% vs 35%). The 4x longer runtime reflects the expanded hyperparameter space and the computational overhead of different quantization kernels.

However, it's worth noting that we only trained for 1 epoch per trial to keep the search tractable. This limited training budget likely disadvantages the more exotic formats like `LinearBinary` and `LinearLog`, which require the model to learn fundamentally different weight representations. With more training epochs, these formats might adapt better and achieve competitive accuracy. Consequently, the high failure rate we observed may partly reflect insufficient training rather than inherent limitations of the formats themselves. Conversely, the formats that performed well (`LinearMinifloatIEEE`, `LinearBlockFP`) are closer to standard floating-point behaviour, making them easier for the model to adapt to quickly.


# References: #

[1] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A Next-generation Hyperparameter Optimization Framework,” arXiv preprint arXiv:1907.10902, 2019.

[2] “Optuna.samplers.tpesampler,” optuna.samplers.TPESampler - Optuna 4.7.0 documentation, https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html (accessed Feb. 4, 2026). 
