# How to use the Statistics Pass Combined with the Ternary Quantiser

## Train a model

We will be using the `toy` model with the `toy-tiny` dataset.
We are collecting data about every layer.

We will be obtaining the maximums and medians.

Train using the following command

`./ch train --config configs/quantized_ops/toy_ternary_collect_stats.toml`

`toy_ternary_collect_stats.toml`:
```toml
# basics
model = "toy"
dataset = "toy-tiny"

[passes.profile_statistics]
by = "type"
target_weight_nodes = ["linear"]
target_activation_nodes = ["relu", "linear"]
num_samples = 32

[passes.profile_statistics.weight_statistics.range_min_max]
dims = "all"
abs = false

[passes.profile_statistics.weight_statistics.range_quantile]
dims = "all"
abs = false
quantile = 0.5

[passes.profile_statistics.activation_statistics.range_min_max]
dims = "all"
abs = false

[passes.profile_statistics.activation_statistics.range_quantile]
dims = "all"
abs = false
quantile = 0.5

[passes.report_node_meta_param]
which = ["software"]

[passes.save_node_meta_param]
save_path = "stats_for_quantisation.toml"

```
## Perform Statistics Pass

`./ch transform --config configs/quantized_ops/toy_ternary_collect_stats.toml`

This outputs the results of the statistics pass to a file named `stats_for_quantisation.toml`.

`stats_for_quantisation.toml`:
```toml
[x.common]
mase_type = "placeholder"
mase_op = "placeholder"

[x.hardware]

[size.common]
mase_type = "implicit_func"
mase_op = "size"

[size.hardware]

[view.common]
mase_type = "implicit_func"
mase_op = "view"

[view.hardware]

[seq_blocks_0.common]
mase_type = "module_related_func"
mase_op = "linear"

[seq_blocks_0.hardware]

[seq_blocks_1.common]
mase_type = "module_related_func"
mase_op = "relu"

[seq_blocks_1.hardware]

[seq_blocks_2.common]
mase_type = "module_related_func"
mase_op = "linear"

[seq_blocks_2.hardware]

[seq_blocks_3.common]
mase_type = "module_related_func"
mase_op = "relu"

[seq_blocks_3.hardware]

[seq_blocks_4.common]
mase_type = "module_related_func"
mase_op = "linear"

[seq_blocks_4.hardware]

[output.common]
mase_type = "output"
mase_op = "output"

[output.hardware]

[x.common.args]

[output.common.results]

[x.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 1, 2, 2,]

[size.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 1, 2, 2,]
from = "x"

[size.common.args.data_in_1]
size = [ 1,]
type = "fixed"
precision = [ 1, 0,]
from = "NA"
value = 0

[size.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1,]
value = 1

[view.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 1, 2, 2,]
from = "x"

[view.common.args.data_in_2]
size = [ 1,]
type = "fixed"
precision = [ 1, 0,]
from = "NA"
value = -1

[view.common.args.data_in_1]
type = "float"
precision = [ 32,]
size = [ 1,]
value = 1
from = "size"

[view.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 4,]

[seq_blocks_0.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 4,]
from = "view"

[seq_blocks_0.common.args.weight]
type = "float"
precision = [ 32,]
size = [ 100, 4,]

[seq_blocks_0.common.args.bias]
type = "float"
precision = [ 32,]
size = [ 100,]

[seq_blocks_0.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]

[seq_blocks_1.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]
from = "seq_blocks_0"

[seq_blocks_1.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]

[seq_blocks_2.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]
from = "seq_blocks_1"

[seq_blocks_2.common.args.weight]
type = "float"
precision = [ 32,]
size = [ 100, 100,]

[seq_blocks_2.common.args.bias]
type = "float"
precision = [ 32,]
size = [ 100,]

[seq_blocks_2.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]

[seq_blocks_3.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]
from = "seq_blocks_2"

[seq_blocks_3.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]

[seq_blocks_4.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 100,]
from = "seq_blocks_3"

[seq_blocks_4.common.args.weight]
type = "float"
precision = [ 32,]
size = [ 2, 100,]

[seq_blocks_4.common.args.bias]
type = "float"
precision = [ 32,]
size = [ 2,]

[seq_blocks_4.common.results.data_out_0]
type = "float"
precision = [ 32,]
size = [ 1, 2,]

[output.common.args.data_in_0]
type = "float"
precision = [ 32,]
size = [ 1, 2,]
from = "seq_blocks_4"

[x.software.results.data_out_0.stat]

[size.software.args.data_in_0.stat]

[view.software.args.data_in_0.stat]

[view.software.args.data_in_1.stat]

[view.software.results.data_out_0.stat]

[seq_blocks_0.software.results.data_out_0.stat]

[seq_blocks_1.software.results.data_out_0.stat]

[seq_blocks_2.software.results.data_out_0.stat]

[seq_blocks_3.software.results.data_out_0.stat]

[seq_blocks_4.software.results.data_out_0.stat]

[output.software.args.data_in_0.stat]

[seq_blocks_0.software.args.data_in_0.stat.range_min_max]
min = -4.972967624664307
max = 4.985265731811523
range = 9.958232879638672
count = 512

[seq_blocks_0.software.args.data_in_0.stat.range_quantile]
min = 0.0833507776260376
max = 0.0833507776260376
range = 0.0
count = 512

[seq_blocks_0.software.args.weight.stat.range_min_max]
min = -0.4987639784812927
max = 0.4998381733894348
range = 0.9986021518707275
count = 400

[seq_blocks_0.software.args.weight.stat.range_quantile]
min = -0.0033772289752960205
max = -0.0033772289752960205
range = 0.0
count = 400

[seq_blocks_0.software.args.bias.stat.range_min_max]
min = -0.49093878269195557
max = 0.4969978332519531
range = 0.9879366159439087
count = 100

[seq_blocks_0.software.args.bias.stat.range_quantile]
min = 0.021730512380599976
max = 0.021730512380599976
range = 0.0
count = 100

[seq_blocks_1.software.args.data_in_0.stat.range_min_max]
min = -6.881089687347412
max = 6.567787170410156
range = 13.448877334594727
count = 12800

[seq_blocks_1.software.args.data_in_0.stat.range_quantile]
min = -0.008840139955282211
max = -0.008840139955282211
range = 0.0
count = 12800

[seq_blocks_2.software.args.data_in_0.stat.range_min_max]
min = 0.0
max = 6.567787170410156
range = 6.567787170410156
count = 12800

[seq_blocks_2.software.args.data_in_0.stat.range_quantile]
min = 0.0
max = 0.0
range = 0.0
count = 12800

[seq_blocks_2.software.args.weight.stat.range_min_max]
min = -0.0999840646982193
max = 0.099988654255867
range = 0.1999727189540863
count = 10000

[seq_blocks_2.software.args.weight.stat.range_quantile]
min = 0.00030127764330245554
max = 0.00030127764330245554
range = 0.0
count = 10000

[seq_blocks_2.software.args.bias.stat.range_min_max]
min = -0.09788546711206436
max = 0.09944488853216171
range = 0.19733035564422607
count = 100

[seq_blocks_2.software.args.bias.stat.range_quantile]
min = -0.010821569710969925
max = -0.010821569710969925
range = 0.0
count = 100

[seq_blocks_3.software.args.data_in_0.stat.range_min_max]
min = -2.998079538345337
max = 2.9003794193267822
range = 5.898458957672119
count = 12800

[seq_blocks_3.software.args.data_in_0.stat.range_quantile]
min = 0.03673841804265976
max = 0.03673841804265976
range = 0.0
count = 12800

[seq_blocks_4.software.args.data_in_0.stat.range_min_max]
min = 0.0
max = 2.9003794193267822
range = 2.9003794193267822
count = 12800

[seq_blocks_4.software.args.data_in_0.stat.range_quantile]
min = 0.03673841804265976
max = 0.03673841804265976
range = 0.0
count = 12800

[seq_blocks_4.software.args.weight.stat.range_min_max]
min = -0.09916079044342041
max = 0.09934879839420319
range = 0.1985095888376236
count = 200

[seq_blocks_4.software.args.weight.stat.range_quantile]
min = 0.003981685731559992
max = 0.003981685731559992
range = 0.0
count = 200

[seq_blocks_4.software.args.bias.stat.range_min_max]
min = -0.07179033756256104
max = 0.0846341997385025
range = 0.15642453730106354
count = 2

[seq_blocks_4.software.args.bias.stat.range_quantile]
min = 0.006421931087970734
max = 0.006421931087970734
range = 0.0
count = 2
```

## Use the `stat-to-conf.py` Script to Generate the new Quantisation Config File

Here, the user can optionally use another config file as a "base" which will be modified to allow for the scaled ternarisation.

We will be using `toy_ternary_scaled.toml` as a base. This means that the values of `quantize.default.config` will be used to create the quantise config for the separate blocks.

It is necessary to have a separate quantisation config for each block as the stats are different for each block, and this was identified as the simplest way to get the appropriate stats to the quantise function.

`toy_ternary_scaled.toml`:
```toml
# basics
model = "toy"
dataset = "toy-tiny"

[passes.quantize]
by = "name"
report = true

[passes.quantize.default.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
weight_scaling_factor = true
weight_width = 2
bias_scaling_factor = true
bias_width = 2

[passes.quantize.seq_blocks_1.config] # ReLU override
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
data_in_frac_width = 0
data_in_mean = "NA"
data_in_median = -0.009714938700199127
data_in_max = 6.568027019500732
```

The script is run in the following way: `./stat-to-conf.py stats_for_quantisation.toml toy_ternary_scaled.toml --base toy_ternary_base.toml`

The written file is:

`toy_ternary_scaled.toml`:
```toml
model = "toy"
dataset = "toy-tiny"

[passes.quantize]
by = "name"
report = true

[passes.quantize.default.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
weight_scaling_factor = true
weight_width = 2
bias_scaling_factor = true
bias_width = 2
data_in_mean = "NA"
data_in_median = 0.0833507776260376
data_in_max = 4.985265731811523
weight_mean = "NA"
weight_median = -0.0033772289752960205
weight_max = 0.4998381733894348
bias_mean = "NA"
bias_median = 0.021730512380599976
bias_max = 0.4969978332519531

[passes.quantize.seq_blocks_1.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
data_in_frac_width = 0
data_in_mean = "NA"
data_in_median = -0.009714938700199127
data_in_max = 6.568027019500732

[passes.quantize.seq_blocks_0.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
weight_scaling_factor = true
weight_width = 2
bias_scaling_factor = true
bias_width = 2
data_in_mean = "NA"
data_in_median = 0.0833507776260376
data_in_max = 4.985265731811523
weight_mean = "NA"
weight_median = -0.0033772289752960205
weight_max = 0.4998381733894348
bias_mean = "NA"
bias_median = 0.021730512380599976
bias_max = 0.4969978332519531

[passes.quantize.seq_blocks_2.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
weight_scaling_factor = true
weight_width = 2
bias_scaling_factor = true
bias_width = 2
data_in_mean = "NA"
data_in_median = 0.0833507776260376
data_in_max = 4.985265731811523
weight_mean = "NA"
weight_median = -0.0033772289752960205
weight_max = 0.4998381733894348
bias_mean = "NA"
bias_median = 0.021730512380599976
bias_max = 0.4969978332519531

[passes.quantize.seq_blocks_3.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
data_in_mean = "NA"
data_in_median = 0.03673841804265976
data_in_max = 2.9003794193267822

[passes.quantize.seq_blocks_4.config]
name = "ternary"
data_in_scaling_factor = true
data_in_width = 2
weight_scaling_factor = true
weight_width = 2
bias_scaling_factor = true
bias_width = 2
data_in_mean = "NA"
data_in_median = 0.0833507776260376
data_in_max = 4.985265731811523
weight_mean = "NA"
weight_median = -0.0033772289752960205
weight_max = 0.4998381733894348
bias_mean = "NA"
bias_median = 0.021730512380599976
bias_max = 0.4969978332519531
```

## Finally, the Quantisation can be Performed in the Usual Way

`./ch transform --config configs/quantized_ops/toy_ternary_scaled.toml --load ../mase_output/toy_classification_toy-tiny_2023-07-27/software/training_ckpts/best.ckpt  --load-type pl`