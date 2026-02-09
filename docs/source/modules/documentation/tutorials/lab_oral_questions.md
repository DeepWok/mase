# Lab Oral Assessment Questions (Prep)

## Lab 0: Foundations (Context Only)

**1)** What is Torch FX and why does MASE use it for graph representation instead of other graph frameworks?

<details>
<summary>Key points for examiner</summary>

- Torch FX: PyTorch-native symbolic tracing framework that creates intermediate representations (IRs) from models
- Benefits: High-level IR, no dedicated runtime needed, every operator correlates to Python object
- Enables transform + optimize + regenerate Python code workflow
- MASE builds MaseGraph on top of Torch FX, adding semantic meaning about workload through metadata
</details>

---


**2)** Explain the difference between analysis passes and transform passes in MASE's pass system.

<details>
<summary>Key points for examiner</summary>

- Analysis passes: Extract info, annotate nodes, generate payloads for subsequent passes (read-only)
- Transform passes: Change graph topology (insert, remove, replace nodes)
- Both iterate over graph nodes with signature `def pass(mg, pass_args={})`
- Examples: `add_common_metadata_analysis_pass` (analysis), `quantize_transform_pass` (transform)
</details>

---


**3)** What is LoRA and why is it considered "parameter efficient"?

<details>
<summary>Key points for examiner</summary>

- LoRA (Low-Rank Adaptation): decomposes weight updates into low-rank matrices
- Standard: `y = X @ W + b`, LoRA: `y = X @ (W + A @ B) + b`
- Freeze the large W, only train A and B (much fewer parameters if rank << hidden_dim)
- Reduces trainable parameters with similar accuracy to full finetuning
- After training, can fuse weights: merge A @ B into W for inference efficiency
</details>

---


**4)** What is the purpose of the `replace_all_uses_with` function before erasing a node in Torch FX?

<details>
<summary>Key points for examiner</summary>

- Before deleting a node, must rewire all downstream nodes that depend on it
- `replace_all_uses_with(parent_node)` updates all users' `args` to point to parent instead
- Without this, erasing leaves downstream nodes with invalid inputs → RuntimeError
- Lab 0 Tutorial 1 task demonstrates: dropout node has 6 users, cannot delete without rewiring
</details>

---

## Lab 1: Quantization and Pruning

### Task 1: QAT vs PTQ Across Bit Widths

**1)** What is the difference between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)?

<details>
<summary>Key points for examiner</summary>

- PTQ: Quantize weights/activations after training completes, no retraining (fast but may lose accuracy)
- QAT: Include fake quantization in training loop, model learns to compensate for quantization noise
    - Fake quantization is when you run the training loop at the original training precision (e.g. FP32) native to the GPU, but limit the values to discrete levels matching the target quantized format (e.g., simulate INT8 precision range stored as FP32 tensors)
- QAT consistently outperforms PTQ. The smaller the bit width, the better the performance increase.
</details>

---


**2)** Why does quantization use both `width` and `frac_width` parameters? What does each control?

<details>
<summary>Key points for examiner</summary>

- Fixed-point representation: total bits split between integer and fractional parts
- `width`: Total number of bits (e.g., 8, 16, 32)
- `frac_width`: Number of fractional bits for decimal precision
- Integer bits = width - frac_width (controls dynamic range)
- Example: Q4.4 = 8 total bits, 4 integer, 4 fractional
- Balance matters: too few frac bits → gradient or value collapse (too many small values get rounded down to zero), too few int bits → overflow
</details>

---


**3)** What function and pass did you use to apply quantization in MASE? What were the key configuration parameters?

<details>
<summary>Key points for examiner</summary>

- Function: `quantize_transform_pass` from MASE passes
- Configuration dict structure: `{"by": "type", "linear": {"config": {...}}}`
- Key parameters: `name` (e.g., "integer"), `data_in_width`, `data_in_frac_width`, `weight_width`, `weight_frac_width`, `bias_width`, `bias_frac_width`
- Varied total widths [4, 6, 8, 12, 16, 24, 32] with both balanced imbalanced precision sweep.
</details>

---


**4)** Describe your experimental procedure for comparing PTQ and QAT. How many epochs did you finetune for QAT?

<details>
<summary>Key points for examiner</summary>

- For each bit width: apply quantization, measure PTQ accuracy immediately
- Then train quantized model for 1 epoch (QAT), measure QAT accuracy
- Tested bit widths: 4, 6, 8, 12, 16, 24, 32
- Same quantization applied to weights, biases, and activations
- FP32 baseline measured first: 0.8140
</details>

---


**5)** Your PTQ accuracy at 4-bit was 0.50 (chance level). What does this tell you about 4-bit precision? Why couldn't QAT recover from this?

<details>
<summary>Key points for examiner</summary>

- 4-bit = 16 representable values (2^4), too few to encode useful information
- Both PTQ and QAT collapse to 0.50 (random guessing on binary sentiment)
- QAT cannot recover because precision floor is too low - no amount of retraining helps
- "4-bit is a failure point" - insufficient representational capacity
- Contrast with 6-bit: QAT recovers to 0.823 (PTQ only 0.689)
</details>

---


**6)** Explain the plot in `quantization_comparison.png`. What are the axes? Describe the key trends for PTQ and QAT curves.

<details>
<summary>Key points for examiner</summary>

- X-axis: bit width (4 to 32), Y-axis: accuracy (0.5 to 0.85)
- PTQ curve: starts at 0.50 (4-bit), rises steeply to 0.81 (16-bit), plateaus
- QAT curve: starts at 0.50 (4-bit), rises to 0.835 (16-bit), plateaus
- Gap largest at low precision (6-bit: +13.4%), narrows at high precision (16+: ~+2%)
- Both plateau above 16 bits (diminishing returns from extra bits) --> because it gets closer to the actual full precision bit width
- QAT consistently above PTQ at all widths except 4-bit
</details>

---


**7)** Your QAT accuracy (0.835) exceeded the FP32 baseline (0.814) at 8+ bits. How is this possible? What phenomenon explains this?

<details>
<summary>Key points for examiner</summary>

- QAT acts as a regularizer: constraining weights to quantized grid during training
- Similar to weight noise injection or dropout - improves generalization
- Model learns more robust features when forced to use limited precision
- Prevents overfitting to training data --> generalizes better to test data
- Same phenomenon observed with pruning at 10% sparsity (0.841 > 0.835)
</details>

---


**8)** Explain the imbalanced precision heatmap. Why do configurations with 2 fractional bits (Q6.2, Q10.2, Q14.2) all fail?

<details>
<summary>Key points for examiner</summary>

- Heatmap shows: rows = total width, columns = frac_width
- All configurations with frac=2 collapse to 0.50 accuracy (PTQ and QAT)
- Fractional bits are critical: model weights need sufficient decimal precision
- Extra integer bits provide no benefit if fractional resolution too low
- BERT weights are small magnitude, need fractional precision to differentiate them (otherwise collapse to zero)
- Contrast: once frac≥4, QAT achieves ~0.833-0.835 regardless of int bit allocation
</details>

---


**9)** According to your imbalanced sweep, PTQ has an "optimal balance point" while QAT is "remarkably robust". Explain this difference with specific examples.

<details>
<summary>Key points for examiner</summary>

- PTQ peaks when frac ≈ width/2: Q4.4 (0.733), Q6.6 (0.809), Q8.8 (0.814)
- PTQ accuracy drops if bits allocated too heavily to int OR frac (cannot adapt)
- QAT: once frac≥4, all configs achieve 0.833-0.835 (Q4.4=0.833, Q8.4=0.833, Q12.4=0.833)
- QAT retrains weights to exploit whatever numerical range available
- PTQ just rounds pre-trained weights → sensitive to representation imbalance
</details>

---


**10)** If you were deploying this model on an 8-bit edge device, would you choose PTQ or QAT? Justify your answer with specific accuracy numbers from your results.

<details>
<summary>Key points for examiner</summary>

- Choose QAT: 8-bit QAT achieves 0.8328 vs PTQ 0.7332 (gap of +9.95%)
- 8-bit QAT nearly matches 16-bit (0.8350), so no need for higher precision
- QAT worth the 1 epoch of retraining cost for 10% accuracy gain
- 8-bit QAT also exceeds FP32 baseline (0.833 > 0.814), proving no quality loss
- Optimal config: Q4.4 (8 total, 4 frac) based on imbalanced sweep
</details>

---


### Task 2: Pruning Strategies

**1)** What is the difference between L1-norm pruning and random pruning? Which weights does each method remove?

<details>
<summary>Key points for examiner</summary>

- L1-norm: Remove weights with smallest absolute values (least contribution to output)
- Random: Remove weights uniformly at random (useful and useless equally)
- L1-norm is magnitude-based structured selection
- Random provides baseline to measure pruning method effectiveness
- L1-norm consistently outperforms random at every sparsity level tested
</details>

---


**2)** What is the difference between structured and unstructured pruning? Which type did you implement?

<details>
<summary>Key points for examiner</summary>

- Structured: Remove entire structures (channels, filters, layers), maintains regular tensor shapes
- Unstructured: Remove individual weights regardless of location, creates sparse matrices
- Structured is hardware-friendly (no sparse support needed)
- Unstructured has higher compression potential but needs specialized hardware
- We implemented unstructured pruning (removed individual weights at arbitrary positions)
</details>

---


**3)** What function did you use to apply pruning? Describe the key configuration parameters.

<details>
<summary>Key points for examiner</summary>

- Function: `prune_transform_pass` from MASE passes
- Configuration: `{"weight": {"sparsity": X, "method": "l1-norm" or "random", "scope": "local"}}`
- Sparsity: fraction of weights to prune (0.1 = 10%, 0.9 = 90%)
- Method: selection strategy (l1-norm vs random)
- Scope: local (per-tensor threshold) vs global (across all tensors)
  - **Local:** Each layer prunes independently to hit the target sparsity
    - Example: Every layer prunes its own smallest 30% of weights
    - Result: All layers have exactly 30% sparsity
  - **Global:** One threshold applied across the entire network
    - Example: Remove all weights < 0.05 anywhere in the network
    - Result: Layers with naturally smaller weights get pruned more aggressively
- We sweep tested sparsity [0.1, 0.2, ..., 0.9] with both methods
</details>

---


**4)** At what sparsity level does random pruning "cliff" occur? What about L1-norm pruning? Explain the difference.

<details>
<summary>Key points for examiner</summary>

- Random cliff: 40% sparsity (drops from 0.763 at 30% to 0.603 at 40%, collapses to 0.515 at 50%)
- L1-norm cliff: 70% sparsity (maintains >0.75 until 70%, drops to 0.607 at 80%)
- 30% point difference in achievable compression ratio
- Random destroys critical computation paths by removing important weights
- L1-norm preserves high-magnitude weights, concentrating useful info in minority of parameters
- BERT-tiny has limited redundancy - cannot tolerate random removal beyond 40%
</details>

---


**5)** Explain the plot in `pruning_comparison.png`. What do you observe about L1-norm at 10% sparsity compared to the baseline?

<details>
<summary>Key points for examiner</summary>

- X-axis: sparsity (0.1 to 0.9), Y-axis: accuracy (0.5 to 0.85)
- Baseline: 0.835 (horizontal reference line)
- L1-norm at 10%: 0.841, EXCEEDS baseline by +0.6%
- This is regularization effect: removing smallest 10% of weights eliminates noise
- Random curve drops monotonically (never exceeds baseline)
- L1-norm stays near baseline until 60% sparsity (0.802), random collapses at 50%
- Both converge near chance (0.5) at 90% sparsity
</details>

---


**6)** Your L1-norm pruning at 30% sparsity achieves 0.832 accuracy. How much does this differ from the baseline (0.835)? What does this tell you about BERT-tiny's weight redundancy?

<details>
<summary>Key points for examiner</summary>

- Difference: 0.832 vs 0.835 = 0.3% drop
- 30% of parameters removed with negligible accuracy loss
- Demonstrates significant weight redundancy in BERT-tiny
- At 50% sparsity, L1-norm still at 0.815 (<2.5% loss)
- Means half the model parameters removable with minimal impact
- Contrast with random: 30% random → 0.763 (8.7% loss)
</details>

---


## Lab 2: Neural Architecture Search

### Task 1: Sampler Comparison

**1)** What is Optuna and why is it useful for Neural Architecture Search?

<details>
<summary>Key points for examiner</summary>

- Optuna: Open-source hyperparameter optimization framework for ML
- Automates the search for optimal hyperparameters
- Provides multiple sampling strategies (Grid, Random, Bayesian optimization like TPE)
- Benefits for NAS: automatically explores architecture search space, tracks trials, finds best configurations
- Alternative to manual hyperparameter tuning or grid search scripts
- Returns best configuration and performance history for analysis
</details>

---


**2)** What is the difference between GridSampler, RandomSampler, and TPESampler in Optuna?

<details>
<summary>Key points for examiner</summary>

- GridSampler: Exhaustive enumeration of all combinations in fixed order (deterministic)
- RandomSampler: Uniform random sampling from search space (no learning)
- TPESampler: Tree-structured Parzen Estimator, a Bayesian optimization approach that models the relationship between hyperparameters and performance, learning from previous trials to suggest promising hyperparameter combinations rather than searching randomly
- Grid infeasible for large spaces, Random can't exploit structure, TPE adapts intelligently
</details>

---


**3)** What is the total size of your architecture search space? How many possible configurations exist?

<details>
<summary>Key points for examiner</summary>

- Search space: num_layers ∈ {2,4,8}, num_heads ∈ {2,4,8,16}, hidden_size ∈ {128,192,256,384,512}, intermediate_size ∈ {512,768,1024,1536,2048}
- Total combinations: 3 × 4 × 5 × 5 = 300 configurations
- Discrete, finite search space
- 100 trials covers 33% of space, sufficient for all samplers to find optimum
- We actually saw both TPE and RandomSampler converge quickly early on -> in larger spaces (e.g., 10^n configs), TPE advantage would be more pronounced
</details>

---


**4)** Describe your model constructor for NAS. How did you use Optuna's `trial.suggest_int` or `trial.suggest_categorical`?

<details>
<summary>Key points for examiner</summary>

- Start with `AutoConfig.from_pretrained(checkpoint)`
- For each hyperparameter: `chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)`
- Sample index, then map to actual value: `setattr(config, param, search_space[param][chosen_idx])`
- Construct model: `AutoModelForSequenceClassification.from_config(config)`
- Train 1 epoch, evaluate on IMDb, return accuracy as objective
</details>

---


**5)** How many trials did you run for each sampler? How long did each trial take (approximately)?

<details>
<summary>Key points for examiner</summary>

- 100 trials per sampler (Grid, Random, TPE)
- Each trial: construct model, train 1 epoch on IMDb, evaluate
- Stored trial number and cumulative best accuracy for plotting
- Used separate Optuna studies for each sampler (fair comparison)
  - Because each Optuna study tracks history of accuracies and internal state for adaptive samplers like TPE
</details>

---


**6)** All three samplers converged to the same best accuracy (0.87232). Why did this happen? Would you expect the same result with a larger search space?

<details>
<summary>Key points for examiner</summary>

- Search space only 300 configs, 100 trials = 33% coverage
- Sufficient for all samplers to hit global optimum eventually
- In larger spaces (continuous or 10^6+ discrete), TPE would maintain advantage
- Grid becomes infeasible at scale (cannot enumerate all)
- Random cannot exploit structure (no acceleration)
- TPE's Bayesian learning shines when optimization is sample-efficiency-critical
</details>

---


**7)** Explain the plot in `sampler_comparison.png`. Which sampler found the best configuration first? On which trial?

<details>
<summary>Key points for examiner</summary>

- X-axis: trial number (1-100), Y-axis: max accuracy achieved so far (cumulative best)
- TPE: found best at trial 2 (0.87232) - fastest
- Random: found best at trial 16
- Grid: found best at trial 27 - slowest (deterministic order)
- All curves plateau at 0.87232 after finding optimum
- TPE curve rises fastest (steep early), Grid rises systematically
</details>

---


**8)** Why does TPESampler find the best configuration so much faster than GridSampler?

<details>
<summary>Key points for examiner</summary>

- TPE learns from trial 1, models which hyperparameters lead to good accuracy
- Concentrates trial 2+ in high-performing regions (adaptive)
- Grid enumerates in fixed order, cannot accelerate based on results
- Grid's "time to best" depends on where optimum falls in enumeration sequence
- Random has non-zero early-hit probability (got lucky at trial 16)
- TPE's Bayesian approach systematically exploits search space structure
</details>

---


### Task 2: Compression-Aware NAS

**1)** What is compression-aware NAS? Why is it better than running NAS first, then compressing the best model afterward?

<details>
<summary>Key points for examiner</summary>

- Compression-aware: include quantization/pruning in each trial's objective function
- Standard approach: NAS → find best uncompressed model → compress it (may not be optimal)
- Different architectures have different quantization/pruning sensitivities
- Compress-aware search finds architectures inherently robust to compression
- Our findings: Compress+Finetune (0.876) > Baseline (0.866) > Compress-Only (0.862)
  - Baseline: no compression
  - Compress-only: quantize without finetuning (evaluate immediately after quantization)
  - Compress+Finetune: train -> quantize -> train 1 more epoch -> evaluate
</details>

---


**2)** Why does post-compression finetuning help? What does the model learn during this phase?

<details>
<summary>Key points for examiner</summary>

- Quantization introduces noise (rounding error), model can adapt weights to compensate
- Finetuning allows model to adjust to quantized representation
- Learns to exploit available numerical range (INT8 grid points)
- Acts as regularizer, can even exceed uncompressed baseline
</details>

---


**3)** Describe your implementation approach for compression-aware search. Did you use the CompressionPipeline or a different method?

<details>
<summary>Key points for examiner</summary>

- Used module-level replacement via `deepsetattr`, NOT CompressionPipeline
  - Need flexibility of manual mode replacement for NAS
  - `deepsetattr` gives more fine-grained control over architecture
  - Apply quantization inside the Optuna objective function
- After training 1 epoch: iterate `named_modules()`, replace `nn.Linear` with `LinearInteger`
- Quantization config: 8-bit fixed point, 4 fractional bits (Q4.4) for data/weights/biases
  - Chose 8-bit because Lab 1 showed good PTQ/QAT balance at this width
  - 4 frac bits prevents collapse
- Three conditions: Baseline (no compression), Compress-Only (quantize + eval), Compress+Finetune (quantize + 1 epoch + eval)
- Used TPESampler (best from Lab 2 Task 1) for 10 trials
</details>

---


**4)** Compare the best accuracy across your three conditions (Baseline, Compress-Only, Compress+Finetune). Which achieved the highest? Why is this surprising?

<details>
<summary>Key points for examiner</summary>

- Baseline: 0.86624 (no compression)
- Compress-Only: 0.86188 (quantize without finetune)
- Compress+Finetune: 0.87644 (quantize + 1 epoch) - HIGHEST
- Surprising because compressed model exceeds uncompressed (0.876 > 0.866)
- Demonstrates quantization as regularizer, finds more generalizable architectures
  - Also because low task complexity (simple binary classification), so compression and quantization act as good regularizers.
  - Same phenomenon as QAT exceeding FP32 baseline in Lab 1
</details>

---


**5)** Explain the plot in `compression_nas_comparison.png`. Describe the variance (spread) of each condition's trials.

<details>
<summary>Key points for examiner</summary>

- X-axis: trial number (1-10), Y-axis: max accuracy so far
- What is not shown in the graph but observed in data:
    - Baseline: mean 0.853, std 0.010 (moderate variance)
    - Compress-Only: mean 0.770, std 0.075 (VERY HIGH variance, range 0.597-0.862)
    - Compress+Finetune: mean 0.872, std 0.002 (remarkably tight, all trials 0.870-0.876)
    - Compress-Only has 40x higher variance than Compress+Finetune
- Finetuning compensates for architecture-dependent quantization sensitivity --> heavily reduces variance
</details>

---


**6)** Why does Compress-Only have such high variance (std=0.075) compared to Compress+Finetune (std=0.002)?

<details>
<summary>Key points for examiner</summary>

- Some architectures tolerate INT8 quantization well, others collapse (0.597 to 0.862 range)
- Quantization noise is architecture-dependent without adaptation
- Compress-Only: model weights unchanged, some configs hit bad numerical range
- Compress+Finetune: retraining adapts weights to quantization, compensates for most sensitivity
- Result: finetuning makes quantization robustness nearly architecture-invariant
</details>

---


**7)** The optimal architecture differs between Baseline and Compress+Finetune conditions. What does this tell you about designing models for deployment?

<details>
<summary>Key points for examiner</summary>

- Baseline best: {num_layers: 0, num_heads: 1, hidden_size: 3, intermediate_size: 4} (larger intermediate)
- Compress+Finetune best: {num_layers: 0, num_heads: 2, hidden_size: 2, intermediate_size: 1} (smaller intermediate)
- Compressed-optimal model uses fewer intermediate features
- Suggests smaller models are more quantization-friendly (less capacity to lose)
- Lesson: cannot optimize for FP32, then compress - need compression-aware search
</details>

---

## Lab 3: Mixed Precision Search

### Task 1: Per-Layer Mixed Precision

**1)** What is mixed precision quantization? Why is it potentially better than uniform precision?

<details>
<summary>Key points for examiner</summary>

- Uniform: all layers use same bit width (e.g., all INT8)
- Mixed precision: different layers can use different precisions (e.g., layer 0 FP32, layer 1 INT8)
- Rationale: some layers more sensitive to quantization than others
- Higher precision where needed, lower precision where tolerable → better accuracy-efficiency tradeoff
</details>

---


**2)** Why is per-layer mixed precision search space much larger than uniform quantization search space?

<details>
<summary>Key points for examiner</summary>

- Uniform: single choice of (type, width, frac_width) applies to all N layers
- Per-layer: independent choice for each of N layers → combinatorial explosion
- Search space: 4 types × multiple width/frac options × 14 layers
- Example: if 10 options per layer, 14 layers → 10^14 configurations
- Makes search much harder (100 trials covers tiny fraction)
</details>

---


**3)** How did you expose per-layer precision as a hyperparameter for Optuna? Describe your model constructor.

<details>
<summary>Key points for examiner</summary>

- Iterate over all Linear layers via `model.named_modules()`
- For each layer: sample quantization type from {Linear, LinearInteger, LinearMinifloatIEEE, LinearLog}
- If quantized type chosen: sample type-specific hyperparameters (e.g., data_in_width ∈ {8,16,32}, frac_width ∈ {2,4,8})
- Use `deepsetattr` to replace `nn.Linear` with sampled quantized layer class
- Each layer gets independent Optuna variables (e.g., `layer.0.attn.self.query_type`)
- Train 1 epoch, return accuracy
</details>

---


**4)** What quantization layer types did you include in your search space? What hyperparameters are specific to each type?

<details>
<summary>Key points for examiner</summary>

- Types: Linear (FP32 baseline), LinearInteger, LinearMinifloatIEEE, LinearLog
- LinearInteger: requires data_in_width, data_in_frac_width, weight_width, weight_frac_width, bias_width, bias_frac_width
- LinearMinifloatIEEE: requires exponent and mantissa bit allocations
- LinearLog: requires base and precision parameters
- Student sampled width ∈ {8,16,32}, frac_width ∈ {2,4,8} for integer types
</details>

---


**5)** Your best per-layer mixed precision accuracy is 0.860. How does this compare to uniform architecture NAS from Lab 2 Task 1 (0.872)? Why the difference?

<details>
<summary>Key points for examiner</summary>

- Mixed precision: 0.860, uniform NAS: 0.872 → 1.4% gap
- Mixed precision underperforms despite per-layer flexibility
- Reason: search space too large (4 types × width options × 14 layers), insufficient trials ran
- TPE has limited data to learn from, cannot find global optimum
- Uniform NAS search space smaller (300 configs), easier to optimize
- More trials would likely close the gap
</details>

---


**6)** Explain the configuration table for your best trial. Which layers are kept at full precision? What pattern do you observe?

<details>
<summary>Key points for examiner</summary>

- Best config keeps 6 of 14 layers at FP32 (full precision)
- Pattern: ALL three attention self-projections (Q, K, V) in layer 0 are FP32
- Layer 0 attention is quantization-sensitive (processes raw token embeddings)
- Layer 1 uses mix of LinearInteger and LinearMinifloatIEEE (more quantization tolerance)
- First encoder layer more critical: distortion propagates through subsequent layers
</details>

---


**7)** Explain the plot in `mixed_precision_search.png`. How many trials collapsed to chance level (0.50)? What is the variance across valid trials?

<details>
<summary>Key points for examiner</summary>

- X-axis: trial number (1-20), Y-axis: max accuracy so far
- 19 valid trials (95%), 1 collapsed to 0.50 (Trial 2, 5%)
- Mean accuracy (valid): 0.8546, std: 0.0042 (tight variance)
- Most mixed-precision configs perform similarly (many "good enough" solutions)
- Global optimum hard to distinguish from near-optimal (flat landscape)
- 5% failure rate expected for this search space (some bad type combos)
</details>

---


### Task 2: Extended Precision Types

**1)** What quantization types did you compare in Task 2? Briefly describe the difference between LinearInteger and LinearMinifloatIEEE.

<details>
<summary>Key points for examiner</summary>

- Types: Linear (FP32), LinearInteger (fixed-point), LinearMinifloatIEEE (low-bit float), LinearLog (logarithmic)
- LinearInteger: fixed-point, bits split between integer and fractional (Q format)
- LinearMinifloatIEEE: floating-point with reduced exponent/mantissa bits (like FP16 but configurable)
- LinearLog: logarithmic representation (sign + log2 magnitude)
- Each has different numerical range and precision tradeoffs
</details>

---


**2)** Why does the best configuration across both Task 1 and Task 2 keep the first encoder layer's attention self-projections at high precision?

<details>
<summary>Key points for examiner</summary>

- Layer 0 processes raw token embeddings (first transformation in encoder)
- Quantization distortion at layer 0 propagates through ALL subsequent layers (compounds)
- Attention self-projections (Q, K, V) compute similarity scores (sensitive operation)
- Layer 1 has already seen one encoder layer, more robust to quantization
- Consistent pattern across Task 1 (FP32 for layer 0 Q/K/V) and Task 2 (high-width for layer 0)
</details>

---


**3)** How did you assign quantization types to each layer in this task? Was it different from Task 1's approach?

<details>
<summary>Key points for examiner</summary>

- Same approach as Task 1: iterate `named_modules()`, sample type + hyperparameters per layer
- Each Linear layer independently assigned one of {Linear, LinearInteger, LinearMinifloatIEEE, LinearLog}
- If quantized type: sample width ∈ {8,16,32}, frac_width ∈ {2,4,8} (for integer types)
- Use `deepsetattr` to replace module in model
- Task 2 adds LinearLog to search space (Task 1 had only Integer/Minifloat)
</details>

---


**4)** Both RandomSampler and TPESampler achieved nearly identical best accuracy (0.86036 vs 0.86052). Why didn't TPE show a larger advantage like it did in Lab 2?

<details>
<summary>Key points for examiner</summary>

- Gap is only 0.00016 (within noise)
- Performance ceiling exists at ~0.860 for BERT-tiny mixed precision on IMDb
- Both samplers hit ceiling, cannot exceed architectural limit
- Random found best at trial 7 (lucky early hit), TPE at trial 37
- TPE had fewer degenerate trials (~4% vs ~8%), showing some learning
- 100 trials covers tiny fraction of space, both samplers limited by sample budget
</details>

---


**5)** Explain the plot in `sampler_comparison_mixed_precision.png`. How many trials collapsed to ~0.5 for each sampler?

<details>
<summary>Key points for examiner</summary>

- X-axis: trial number (1-100), Y-axis: max accuracy so far
- Random: ~8 trials at 0.5 (8% failure rate)
- TPE: ~4 trials at 0.5 (4% failure rate)
- TPE learns from failures, avoids configs that previously collapsed
- Random cannot adapt, continues sampling bad regions with uniform probability
- Both curves plateau at ~0.860 (performance ceiling)
</details>

---


**7)** Mixed precision (0.860) underperforms uniform NAS (0.872) across both Task 1 and Task 2. Should we abandon mixed precision? Why or why not?

<details>
<summary>Key points for examiner</summary>

- Do NOT abandon: mixed precision offers finer control and layer-specific optimisation, useful for hardware constraints
- Underperformance due to search difficulty, not approach limitation
- Need more trials (1000+) or better search algorithms (evolutionary, RL)
</details>

---

## Lab 4: Hardware Stream

**1)** What is the purpose of the hardware metadata pass (`add_hardware_metadata_analysis_pass`)? How does it differ from software metadata?

<details>
<summary>Key points for examiner</summary>

- Hardware metadata: encodes how node is realized in hardware (storage type, signal naming, parameter expansion)
- Required for static analysis, transformation, RTL generation
- Software metadata: assumes dynamic execution, flexible memory, runtime scheduling (no explicit signals)
- Hardware metadata includes: verilog_param (compile-time params), interface (BRAM/transpose), dependence_files, toolchain (INTERNAL/EXTERNAL/HLS)
- Bridges gap between tensor-level semantics and signal-level hardware interfaces
</details>

---


**2)** Describe the pipeline structure of the generated `top.sv` module. What are the two stages? What operations does each perform?

<details>
<summary>Key points for examiner</summary>

- Two-stage pipeline: fixed_linear (fc1_inst) → fixed_relu
- Stage 1 (fixed_linear): implements Linear(4, 8) with quantized matrix multiplication (input @ weights + bias)
- Stage 2 (fixed_relu): ReLU activation (max(0, x)), purely combinational
- Quantization: all tensors use Q5.3 (8-bit fixed-point, 5 integer, 3 fractional bits)
- Parallelism: accepts 4 inputs in parallel, produces 4 of 8 outputs per cycle (two output tiles)
</details>

---


**3)** How are the model's trained weights (fc1.weight and fc1.bias) provided to the hardware? What modules handle this?

<details>
<summary>Key points for examiner</summary>

- `fc1_weight_source` and `fc1_bias_source` modules provide constant parameters
- These are ROM-like modules storing quantized weights/biases from trained PyTorch model
- Connected via handshake interface (valid/ready) to linear block
- Stream parameter data tile-by-tile to fc1 during computation
- With parallelism=4, output size=8, fc1 streams weights for two output tiles sequentially
</details>

---


**4)** Examine the GTKWave waveform. Why does `relu_data_out_0_valid` rise on the exact same clock edge as `data_out_0_valid` (fc1 output)?

<details>
<summary>Key points for examiner</summary>

- Confirms ReLU is purely combinational logic (zero pipeline latency)
- No register between fc1 output and relu output
- ReLU function resolved within same clock cycle as fc1 produces data
- Combinational path: fc1_output → ReLU comparator/mux → relu_output
- Pipeline depth dominated by fc1 matrix multiply, ReLU adds no cycles
</details>

---


**5)** Explain the waveform in `gtkwave_screenshot.png`. Trace the data flow from input handshake to output handshake.

<details>
<summary>Key points for examiner</summary>

- Input handshake: `data_in_0_valid` asserted by testbench, `data_in_0_ready` already high → fc1 accepts [0x01, 0x00, 0x06, 0x03]
- fc1 computes: streams weights/biases via `fc1_weight_ready/valid`, `fc1_bias_ready/valid` handshakes
- fc1 output: `data_out_0_valid` goes high when computation done (~280ns)
- ReLU output: `relu_data_out_0_valid` rises same cycle (combinational)
- Downstream backpressure: `relu_data_out_0_ready` controls when monitor accepts output
- Total pipeline: input → fc1 (dominates latency) → relu (0 cycles) → output
</details>

---


**6)** Why does the `fixed_linear` stage dominate the pipeline latency? How many output tiles does it compute sequentially?

<details>
<summary>Key points for examiner</summary>

- Matrix multiplication requires streaming weights and computing dot products
- Parallelism=4, output size=8 → produces 4 outputs per tile
- Two output tiles computed sequentially: tile 0 (outputs 0-3), tile 1 (outputs 4-7)
- Each tile: stream weights from BRAM, compute 4 dot products, accumulate with bias
- ReLU is single comparator (max(0,x)), negligible delay compared to multiply-accumulate
</details>

---


**7)** For the Leaky ReLU extension, what additional parameters did you add to the module instantiation? What do they represent?

<details>
<summary>Key points for examiner</summary>

- Added three parameters: `NEGATIVE_SLOPE_PRECISION_0`, `NEGATIVE_SLOPE_PRECISION_1`, `NEGATIVE_SLOPE_VALUE`
- PRECISION_0: integer bits for slope encoding
- PRECISION_1: fractional bits for slope encoding
- NEGATIVE_SLOPE_VALUE: the actual slope coefficient (0.125 = 1/8 = 1/2^3 in Q8.3)
- Leaky ReLU output: x if x>0, else (NEGATIVE_SLOPE_VALUE * x) >>> PRECISION_1
- Uses fixed-point multiplier and arithmetic right-shift for negative region
</details>

---


**8)** Compare ReLU vs Leaky ReLU simulation results. Did latency change? Why or why not?

<details>
<summary>Key points for examiner</summary>

- Simulation time: ReLU 30.09s, Leaky ReLU 29.92s (identical within noise)
- Both purely combinational (0 clock cycles of latency)
- Leaky ReLU adds multiplier + right-shift (combinational logic), no registers
- Pipeline depth unchanged: still dominated by fixed_linear stage
- Functional difference: ReLU zeroes negatives, Leaky ReLU scales by 0.125
</details>

---


**9)** Why is Leaky ReLU considered an improvement over ReLU for training? Does this matter for inference-only hardware?

<details>
<summary>Key points for examiner</summary>

- ReLU zeros all negative values → "dying ReLU" problem (neuron stops learning if always negative)
- Leaky ReLU preserves scaled gradient for negatives → prevents dead neurons during training
- For inference-only hardware: less critical, model already trained
- Functional difference: negative activations propagate (scaled) through subsequent layers
- Area cost minimal: single multiplier + shifter for parallelism=4
- May improve model robustness to input distribution shift at inference
</details>

---

## Cross-Lab Synthesis Questions

**1)** A consistent theme across Lab 1 QAT, Lab 1 Pruning at 10%, and Lab 2 Compress+Finetune is that compressed models can exceed uncompressed baselines. What common mechanism explains this phenomenon?

<details>
<summary>Key points for examiner</summary>

- All three involve constrained optimization: QAT, pruning, compress+finetune
- Constraints act as regularizers, preventing overfitting to training data
- Model forced to learn more robust features when capacity limited
- Similar to dropout, weight decay, or noise injection
- Demonstrates compression is not purely destructive - can improve generalization
- Sweet spot exists: too much compression destroys capacity (e.g. 4-bit collapse, 90% sparsity)
</details>

---


**2)** Lab 2 Task 1 successfully searched a 300-configuration space with 100 trials, while Lab 3 mixed precision struggled despite 100 trials. What does this tell you about the relationship between search space dimensionality and optimization difficulty?

<details>
<summary>Key points for examiner</summary>

- Lab 2 uniform NAS: 3×4×5×5=300 configs, 100 trials = 33% coverage → all samplers found optimum (0.872)
- Lab 3 mixed precision: 4 types × width options × 14 layers = exponential space, 100 trials = tiny fraction → ceiling at 0.860
- High-dimensional spaces need exponentially more samples (curse of dimensionality)
- TPE's adaptive advantage matters more in large spaces, but still limited by sample budget
- Uniform quantization easier to optimize (fewer hyperparameters, shared across layers)
- Mixed precision needs 1000+ trials or hierarchical search (optimize layer groups, not all independently)
</details>

---


**3)** Trace the software-to-hardware pipeline from Lab 1-3 quantization decisions to Lab 4 hardware implementation. How do the bit width choices (e.g., 8-bit, Q4.4) from software experiments inform the RTL parameters?

<details>
<summary>Key points for examiner</summary>

- Lab 1 findings: 8-bit with 4 frac bits (Q4.4) achieves good PTQ/QAT balance, frac≥4 required
- Lab 2-3: used 8-bit INT8 for compression-aware search (practical constraint)
- Lab 4 hardware: uses Q5.3 (8-bit total, 5 int, 3 frac) for fixed_linear and weights
- RTL parameters: `DATA_IN_0_PRECISION_0` (int bits), `DATA_IN_0_PRECISION_1` (frac bits)
- Software experiments validate numerical range needed → hardware generation uses these constraints
- Quantization-aware training produces models tolerant to fixed-point hardware
- Without Lab 1-3: would blindly generate hardware, risk accuracy collapse
- Pipeline: PyTorch QAT → quantized .pt checkpoint → MASE hardware metadata → Verilog emission
</details>

---