import os
from functools import partial

from chop.nn.quantized.functional.linear import (
    linearBinary,
    linearBinaryScaling,
    linearBlockFP,
    linearBlockLog,
    linearBlockMinifloat,
    linearInteger,
    linearLog,
    linearMXIntHardware,
    linearMinifloatDenorm,
    linearMinifloatIEEE,
    linearTernary,
)
import torch
from torch import Tensor
from torch.nn import functional as F


from ..utils import get_stats, quantiser_passthrough

from chop.nn.quantizers import (
    residual_sign_quantizer,
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    integer_floor_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
    mxint_hardware,
    mxint_quantizer,
    mxfp_quantizer,
)

# `chop.nn.quantizers.rotation` triggers an `import fast_hadamard_transform`
# at load time. We defer that import to RotateMXIntLinear.forward so users
# that never touch the rotate path don't need the CUDA extension installed.

from chop.nn.quantized.modules.phase_context import get_runtime_phase
from chop.nn.quantized.modules.phase_config import (
    GPTQ_DECODE_WEIGHT_ATTR,
    normalize_phase_q_config,
    resolve_module_phase_config,
)

# LUTNet
import numpy as np
from typing import Type
from chop.nn.quantizers.LUTNet.BaseTrainer import BaseTrainer, LagrangeTrainer
from chop.nn.quantizers.LUTNet.MaskBase import MaskBase, MaskExpanded

# LogicNets
from chop.nn.quantizers.LogicNets.utils import (
    generate_permutation_matrix,
    get_int_state_space,
    fetch_mask_indices,
)

# LogicNets
from chop.nn.quantizers.LogicNets.utils import (
    generate_permutation_matrix,
    get_int_state_space,
    fetch_mask_indices,
)


class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )
        self.bypass = False
        self.pruning_masks = None
        # NOTE: Quantizers properties are not needed for now
        # self.x_quantizer = None
        # self.w_quantizer = None
        # self.b_quantizer = None
        # self.out_quantizer = None

    # NOTE: This is not needed for now
    # def forward(self, x: Tensor) -> Tensor:
    #     if self.bypass:
    #         # if bypass, there is no quantization
    #         return F.linear(x, self.weight, self.bias)
    #     else:
    #         x = self.x_quantizer(x)
    #         w = self.w_quantizer(self.weight)
    #         bias = self.b_quantizer(self.bias) if self.bias is not None else None
    #         out = F.linear(x, w, bias)
    #         if self.out_quantizer is None:
    #             return out
    #         return self.out_quantizer(out)


class LinearInteger(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
        out_config=None,
        floor=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        # Add floor attribute to the config
        if floor is not None:
            config["floor"] = floor
        self.config = config
        self.out_config = out_config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearInteger(x, self.weight, self.bias, self.config, self.out_config)


class LinearMinifloatDenorm(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearMinifloatDenorm(x, self.weight, self.bias, self.config)


class LinearMinifloatIEEE(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearMinifloatIEEE(x, self.weight, self.bias, self.config)


class LinearLog(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearLog(x, self.weight, self.bias, self.config)


class LinearBlockFP(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearBlockFP(x, self.weight, self.bias, self.config)


class LinearBlockMinifloat(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearBlockMinifloat(x, self.weight, self.bias, self.config)


class LinearBlockLog(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearBlockLog(x, self.weight, self.bias, self.config)


class LinearBinary(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearBinary(x, self.weight, self.bias, self.config)


class LinearBinaryScaling(_LinearBase):
    """
    Binary scaling variant of the linear transformation layer.

        - "bypass": Bypass quantization for standard linear transformation.
        - "data_in_stochastic", "bias_stochastic", "weight_stochastic": Stochastic settings.
        - "data_in_bipolar", "bias_bipolar", "weight_bipolar": Bipolar settings.
        - "binary_training": Apply binary scaling during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        # self.gamma = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearBinaryScaling(x, self.weight, self.bias, self.config)


class LinearTernary(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

        w_scaling_factor = config["weight_scaling_factor"]
        w_mean = get_stats(config, "weight_mean")
        w_median = get_stats(config, "weight_median")
        w_max = get_stats(config, "weight_max")
        self.w_quantizer = partial(
            ternary_quantizer,
            scaling_factor=w_scaling_factor,
            maximum=w_max,
            median=w_median,
            mean=w_mean,
        )
        self.x_quantizer = quantiser_passthrough
        self.b_quantizer = quantiser_passthrough
        # self.b_quantizer = partial(
        #     ternary_quantizer,
        #     scaling_factor=b_scaling_factor,
        #     maximum=b_max,
        #     median=b_median,
        #     mean=b_mean,
        # )

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearTernary(x, self.weight, self.bias, self.config)


# LUT
class LinearBinaryResidualSign(_LinearBase):
    """
    Binary Linear layer with redisual sign variant of the linear transformation layer.

        - "bypass": Bypass quantization for standard linear transformation.
        - "data_in_stochastic", "bias_stochastic", "weight_stochastic": Stochastic settings.
        - "data_in_bipolar", "bias_bipolar", "weight_bipolar": Bipolar settings.
        - "binary_training": Apply binary scaling during training.
        - "data_in_levels": The num of residual layers to use.
        - "data_in_residual_sign" : Apply residual sign on input
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.levels = config.get("data_in_levels", 2)  # NOTE: Hardcode 2 for now
        self.config = config
        self.bypass = config.get("bypass", False)
        self.gamma = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # Initialized parameter
        ars = np.arange(self.levels) + 1.0
        ars = ars[::-1]
        means = ars / np.sum(ars)
        # Create a torch.nn.Parameter from the means tensor
        self.means = (
            torch.nn.Parameter(
                torch.tensor(means, dtype=torch.float32, requires_grad=True)
            )
            if self.config.get("data_in_residual_sign", True)
            else None
        )
        # prunning masks
        self.pruning_masks = torch.nn.Parameter(
            torch.ones_like(self.weight), requires_grad=False
        )

        if self.bypass:
            return
        x_stochastic, w_stochastic = (
            config["data_in_stochastic"],
            config["weight_stochastic"],
        )
        x_bipolar, w_bipolar = (
            config["data_in_bipolar"],
            config["weight_bipolar"],
        )

        self.binary_training = config["binary_training"]

        self.w_quantizer = partial(
            binary_quantizer, stochastic=w_stochastic, bipolar=w_bipolar
        )
        self.x_quantizer = partial(
            binary_quantizer, stochastic=x_stochastic, bipolar=x_bipolar
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)

        x_expanded = 0
        if self.means is not None:
            out_bin = residual_sign_quantizer(
                levels=self.levels, x_quantizer=self.x_quantizer, means=self.means, x=x
            )
            for l in range(self.levels):
                x_expanded = x_expanded + out_bin[l, :, :]
        else:
            x_expanded = x

        if self.binary_training:
            w = self.w_quantizer(self.weight)
            return F.linear(
                x_expanded,
                w * self.gamma.abs() * self.pruning_masks,
                self.bias,
            )
        else:
            self.weigh = self.weight.data.clamp_(-1, 1)
            return F.linear(
                x_expanded,
                self.weight * self.gamma.abs() * self.pruning_masks,
                self.bias,
            )


class LinearLUT(torch.nn.Module):
    input_mask: torch.Tensor
    tables_count: int
    in_features: int
    out_features: int
    trainer: BaseTrainer
    mask_builder_type: Type[MaskBase]
    mask_builder: MaskBase

    def __init__(
        self,
        config: None,
        in_features: int,
        out_features: int,
        mask_builder_type: Type[MaskBase] = MaskExpanded,
        trainer_type: Type[BaseTrainer] = LagrangeTrainer,
        bias: bool = True,
        device: str = None,
    ) -> None:
        super(LinearLUT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.levels = config.get("data_in_levels", 2)
        self.input_expanded = config["data_in_input_expanded"]
        self.k = config["data_in_k"]
        self.kk = 2 ** config["data_in_k"]
        self.mask_builder_type = mask_builder_type
        # Initialize mask builder
        self.input_mask = self._input_mask_builder()
        # TODO: table * output feature map
        self.tables_count = self.mask_builder.get_tables_count() * self.out_features
        self.trainer = trainer_type(
            levels=self.levels,
            tables_count=self.tables_count,
            k=config["data_in_k"],
            binarization_level=(1 if config["data_in_binarization_level"] == 1 else 0),
            input_expanded=config["data_in_input_expanded"],
            device=device,
        )
        self.weight = self.trainer.weight
        self.pruning_masks = self.trainer.pruning_masks
        # TODO: we might need to this later on
        # stdv = 1 / np.sqrt(self.in_features)
        # w = np.random.normal(loc=0.0, scale=stdv, size=list(self.trainer.weight.shape)).astype(np.float32)
        # self.trainer.weight = torch.nn.Parameter(
        #    torch.tensor(w, requires_grad=True))

        self.bias = (
            torch.nn.Linear(1, out_features, device=device).bias if bias else None
        )

        # Residual sign code
        self.x_quantizer = partial(binary_quantizer, stochastic=False, bipolar=True)
        ars = np.arange(self.levels) + 1.0
        ars = ars[::-1]
        means = ars / np.sum(ars)
        self.means = torch.nn.Parameter(
            torch.tensor(means, dtype=torch.float32, requires_grad=True)
        )

    def _table_input_selections_builder(self) -> np.array:
        _all_inputs_set = set(range(self.in_features))
        result = []
        for in_idx in range(self.in_features):
            _idx_set = set([in_idx])
            _selection = list(_all_inputs_set - _idx_set)
            result.append((in_idx, _selection))
        return result

    def _input_mask_builder(self) -> torch.Tensor:
        """
        Initializing table (using indices for the connections)
        """
        result = []
        # TODO: elements can appear more than once in the feature-1 input?
        for _ in range(self.out_features):
            self.mask_builder = self.mask_builder_type(
                self.k, self._table_input_selections_builder(), True
            )
            result.append(self.mask_builder.build())
        return np.concatenate(result)

    def forward(
        self,
        input: torch.Tensor,
        targets: torch.tensor = None,
        initalize: bool = False,
    ):
        assert len(input.shape) == 2
        batch_size = input.shape[0]
        out_bin = residual_sign_quantizer(
            levels=self.levels, x_quantizer=self.x_quantizer, means=self.means, x=input
        )
        expanded_input = out_bin[:, :, self.input_mask]  # [levels, batch_size, mask]

        output = self.trainer(expanded_input, targets, initalize).squeeze()
        output = output.view(batch_size, -1)
        assert output.shape[-1] == self.tables_count
        output = output.view(
            batch_size,
            self.out_features,
            int(self.tables_count / self.out_features),
        )
        output = output.sum(-1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def pre_initialize(self):
        self.trainer.clear_initializion()

    def update_initialized_weights(self):
        self.trainer.update_initialized_weights()


class LinearLogicNets(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
        activation_module=None,  # To initialize a LogicNets, activation functions are needed
        input_layers=None,  # A LogicNets layer may be merged with one or more inputs layers such as activations and batchnorm
        output_layers=None,  # A LogicNets layer may be merged with one or more output layers such as activations and batchnorm
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        self.x_width, self.x_frac_width = (
            config["data_in_width"],
            config["data_in_frac_width"],
        )
        self.y_width, self.y_frac_width = (
            config["data_out_width"],
            config["data_out_frac_width"],
        )

        self.x_quantizer = partial(
            integer_quantizer, width=self.x_width, frac_width=self.x_frac_width
        )
        self.y_quantizer = partial(
            integer_quantizer, width=self.y_width, frac_width=self.y_frac_width
        )

        # self.input_quant = input_quant
        # self.output_quant = output_quant
        self.activation = activation_module
        self.is_lut_inference = True
        self.neuron_truth_tables = None
        # self.calculate_truth_tables()
        # self.apply_input_quant = apply_input_quant
        # self.apply_output_quant = apply_output_quant
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.apply_layers = False

    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(
        self,
        connected_input: Tensor,
        input_perm_matrix: Tensor,
        bin_output_states: Tensor,
    ) -> Tensor:
        fan_in_size = connected_input.shape[1]
        ci_bcast = connected_input.unsqueeze(2)  # Reshape to B x Fan-in x 1
        pm_bcast = input_perm_matrix.t().unsqueeze(
            0
        )  # Reshape to 1 x Fan-in x InputStates
        eq = (ci_bcast == pm_bcast).sum(
            dim=1
        ) == fan_in_size  # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )
        indices = torch.argmax(eq.type(torch.int64), dim=1)
        return bin_output_states[indices]

    def lut_forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(
            x, 1
        )  # N - added this; is 1 needed to flatten all dims except batch?
        # if self.apply_input_quant:
        #     x = self.input_quant(x) # Use this to fetch the bin output of the input, if the input isn't already in binary format
        x = self.encode(self.x_quantizer(x))
        y = torch.zeros((x.shape[0], self.out_features))
        # Perform table lookup for each neuron output
        for i in range(self.out_features):
            indices, input_perm_matrix, bin_output_states = self.neuron_truth_tables[i]
            # Move logicnets tensor to GPU
            input_perm_matrix = input_perm_matrix.to(x.device)
            bin_output_states = bin_output_states.to(x.device)
            connected_input = x[:, indices]
            y[:, i] = self.table_lookup(
                connected_input, input_perm_matrix, bin_output_states
            )
        return y

    def construct_mask_index(self):
        # contract a mask have the same shape as self.weight but with zero element being assign to zero and other assign to 1
        self.mask = torch.where(
            self.weight != 0, torch.tensor(1), torch.tensor(0)
        ).reshape(
            self.weight.shape[0], -1
        )  # pay attention to dimension (out_feature, in_feature)

    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables(self):
        # print(
        #     "weight", torch.where(self.weight != 0, torch.tensor(1), torch.tensor(0))
        # )  # pay attention to dimension (out_feature, in_feature)
        with torch.no_grad():
            # Precalculate all of the input value permutations
            input_state_space = list()  # TODO: is a list the right data-structure here?
            bin_state_space = list()
            # get a neuron_state
            for m in range(self.in_features):
                neuron_state_space = self.decode(
                    get_int_state_space(self.x_width)
                )  # TODO: this call should include the index of the element of interest
                bin_space = get_int_state_space(
                    self.x_width
                )  # TODO: this call should include the index of the element of interest
                input_state_space.append(neuron_state_space)
                bin_state_space.append(bin_space)

            neuron_truth_tables = list()
            self.construct_mask_index()  # construct pruning mask
            for n in range(self.out_features):
                input_mask = self.mask[
                    n, :
                ]  # N: select row of mask tensor that corresponds to the output feature on this iteration
                fan_in = torch.sum(input_mask)
                indices = fetch_mask_indices(input_mask)
                # Generate a matrix containing all possible input states
                input_permutation_matrix = generate_permutation_matrix(
                    [input_state_space[i] for i in indices]
                )
                bin_input_permutation_matrix = generate_permutation_matrix(
                    [bin_state_space[i] for i in indices]
                )
                # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
                num_permutations = input_permutation_matrix.shape[0]
                padded_perm_matrix = torch.zeros((num_permutations, self.in_features))
                padded_perm_matrix[:, indices] = input_permutation_matrix

                bin_output_states = self.encode(self.math_forward(padded_perm_matrix))[
                    :, n
                ]  # Calculate bin for the current input

                # Append the connectivity, input permutations and output permutations to the neuron truth tables
                neuron_truth_tables.append(
                    (indices, bin_input_permutation_matrix, bin_output_states)
                )  # Change this to be the binary output states
        self.neuron_truth_tables = neuron_truth_tables

    def math_forward(self, input: Tensor) -> Tensor:
        if self.activation == "unittest":
            # This is the for performing unittest on the layer
            return self.y_quantizer(
                F.linear(self.x_quantizer(input), self.weight, self.bias)
            )

        if self.apply_layers:
            x = input
            if self.input_layers:
                x = self.run_layers(x, self.input_layers)

            y = self.y_quantizer(F.linear(self.x_quantizer(x), self.weight, self.bias))

            if self.output_layers:
                y = self.run_layers(y, self.output_layers)
            return y

        # This is the case where the linear layer is the only module in the LogicNets module
        return self.y_quantizer(
            F.linear(self.x_quantizer(input), self.weight, self.bias)
        )

    def set_fused(self, fused: bool):
        self.apply_layers = fused

    def run_layers(self, input: Tensor, layers) -> Tensor:
        assert isinstance(layers, list)
        y = input
        for layer in layers:
            layer_name = layer.__class__.__name__
            SUPPORTED_LAYERS = {
                "ReLU": 1,
                "Tanh": 1,
                "BatchNorm1d": 1,
                "str": 0,
            }  # "str" type is short the "output". Hence this logicnets will be a pure linear without activation.
            if layer_name not in SUPPORTED_LAYERS:
                raise ValueError(
                    "Unsupported output layer {}. Please choose from {}".format(
                        layer_name, list(SUPPORTED_LAYERS.keys())
                    )
                )
            if SUPPORTED_LAYERS[layer_name]:
                y = layer(y)
        return y

    def encode(self, input: Tensor) -> Tensor:
        return input * 2**self.x_frac_width

    def decode(self, input: Tensor) -> Tensor:
        return input / 2**self.x_frac_width

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            return self.decode(self.lut_forward(x))
        else:
            return self.math_forward(x)


class LinearMXIntHardware(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
        out_config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.out_config = out_config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return linearMXIntHardware(
            x, self.weight, self.bias, self.config, self.out_config
        )

class _PhaseAwareMXLinearBase(_LinearBase):
    """Shared machinery for phase-aware MX-format linear layers.

    Bank layout (decode-side disaggregated serving):
    - ``self.weight`` is the PREFILL bank. With a bypassed prefill bucket it
      stays FP (the prefill chip is unquantised); with a quantising prefill
      bucket it is quantised in place (legacy behaviour).
    - ``_decode_weight_q`` / ``_decode_bias_q`` hold the DECODE bank when the
      decode bucket quantises weights differently from prefill. It is built
      from the FP source weights (never from an already-quantised prefill
      bank), or installed externally by the GPTQ pre-pass
      (``gptq_config["phase"] = "decode"``) via ``adopt_decode_gptq_weight``.
    - ``_decode_weight_fp`` / ``_decode_bias_fp`` hold an FP snapshot for the
      inverse deployment (quantised prefill + ``fp_only`` decode), where the
      in-place prefill quantisation would otherwise destroy the FP weights
      decode needs.

    When both phase buckets are identical (any legacy flat config) no decode
    bank is allocated and both phases share ``self.weight`` — no extra memory
    and unchanged single-bank behaviour. With two distinct banks, both are
    stored at the model dtype (fake quantisation), so weight memory doubles —
    the cost of emulating two chips in one process.

    Subclasses provide the format-specific ``_quantize_weight_with_config``
    and ``_quantize_activation_with_config`` hooks; the Rotate variants only
    override the activation hook.
    """

    # NOTE: backward is not supported — inference only (PTQ)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self._init_phase_state(config)
        self._register_phase_bank_buffers()

    def _init_phase_state(self, config: dict) -> None:
        # Raw config retained for zero-copy class swaps (rotation search).
        self.config = config
        self.phase_q_config = normalize_phase_q_config(config)
        self.decode_policy = self.phase_q_config["decode_policy"]
        self.prefill_config = resolve_module_phase_config(self.phase_q_config, "prefill")
        self.decode_config = resolve_module_phase_config(self.phase_q_config, "decode")
        # Legacy attribute contract: these mirror the prefill bucket because
        # ``self.weight`` is the prefill bank.
        self.bypass = self.prefill_config.get("bypass", False)
        self.gptq = self.prefill_config.get("gptq", False)
        self.clip_search = self.prefill_config.get("clip_search", False)
        self.decode_gptq = self.decode_config.get("gptq", False)
        # Single-bank fast path: identical buckets share ``self.weight``.
        self.shared_phase_banks = self.prefill_config == self.decode_config
        self._validate_phase_weight_configs()

    def _register_phase_bank_buffers(self) -> None:
        # Non-persistent: decode banks are runtime state rebuilt from the FP
        # source, so checkpoints stay upstream-compatible.
        self.register_buffer("_decode_weight_q", torch.empty(0), persistent=False)
        self.register_buffer("_decode_bias_q", torch.empty(0), persistent=False)
        self.register_buffer("_decode_weight_fp", torch.empty(0), persistent=False)
        self.register_buffer("_decode_bias_fp", torch.empty(0), persistent=False)

    # ---- format-specific hooks -------------------------------------------
    # Config keys a bucket must fully provide to quantise weights / bias.
    _WEIGHT_CFG_KEYS: tuple = ()
    _BIAS_CFG_KEYS: tuple = ()

    def _quantize_weight_with_config(self, w: Tensor, cfg: dict, block_dim: int) -> Tensor:
        raise NotImplementedError

    def _weight_config_present(self, cfg: dict) -> bool:
        return all(cfg.get(k) is not None for k in self._WEIGHT_CFG_KEYS)

    def _bias_config_present(self, cfg: dict) -> bool:
        return all(cfg.get(k) is not None for k in self._BIAS_CFG_KEYS)

    def _quantize_bias_with_config(self, b: Tensor, cfg: dict) -> Tensor:
        raise NotImplementedError

    def _quantize_activation_with_config(self, x: Tensor, cfg: dict) -> Tensor:
        """Quantise the activation by the keys PRESENT, not the weight format.

        The module class is chosen by the WEIGHT format, but mixed-format
        deployments (MXINT weights + MXFP activations and vice versa) put the
        other format's ``data_in_*`` keys in the config. Dispatching on the
        keys keeps that legal; a class that only read its own format's keys
        would silently skip activation quantisation. No ``data_in`` keys at
        all means the activation is intentionally unquantised.
        """
        block_size = cfg.get("data_in_block_size")
        if block_size is None:
            return x
        if cfg.get("data_in_width") is not None:
            return self._quantize_activation_mxint(x, block_size, cfg)
        if (cfg.get("data_in_exponent_width") is not None
                and cfg.get("data_in_frac_width") is not None):
            return self._quantize_activation_mxfp(x, block_size, cfg)
        return x

    def _quantize_activation_mxint(self, x: Tensor, block_size: int, cfg: dict) -> Tensor:
        # mxint_quantizer natively handles DTensor inputs (TP-compatible).
        return mxint_quantizer(
            x, block_size=block_size, element_bits=cfg["data_in_width"], block_dim=-1,
        )

    def _quantize_activation_mxfp(self, x: Tensor, block_size: int, cfg: dict) -> Tensor:
        return mxfp_quantizer(
            x, block_size=block_size,
            element_exp_bits=cfg["data_in_exponent_width"],
            element_frac_bits=cfg["data_in_frac_width"],
            block_dim=-1,
        )

    def _validate_phase_weight_configs(self) -> None:
        """Reject partial weight configs.

        A bucket may provide all weight keys (quantise) or none (weights stay
        FP, activation-only quantisation). Anything in between is almost
        certainly a typo and would otherwise silently skip quantisation.
        """

        for phase_name, cfg in (
            ("prefill", self.prefill_config),
            ("decode", self.decode_config),
        ):
            if cfg.get("bypass", False) or cfg.get("gptq", False):
                continue
            present = [k for k in self._WEIGHT_CFG_KEYS if cfg.get(k) is not None]
            if present and len(present) != len(self._WEIGHT_CFG_KEYS):
                missing = [k for k in self._WEIGHT_CFG_KEYS if cfg.get(k) is None]
                raise ValueError(
                    f"{self.__class__.__name__}: incomplete weight config in "
                    f"the {phase_name} bucket — found {present}, missing "
                    f"{missing}. Provide all weight keys, or none for "
                    "activation-only quantisation."
                )

    # ---- bank construction ------------------------------------------------
    @torch.no_grad()
    def _quantize_weight_banked(self, w: Tensor, cfg: dict, block_dim: int) -> Tensor:
        """Chunked (and optionally offloaded) weight quantisation for bank builds
        """
        if w.ndim != 2 or block_dim not in (1, -1):
            return self._quantize_weight_with_config(w, cfg, block_dim)
        route = os.environ.get("MASE_PHASE_BANK_DEVICE") or None
        if route is not None and str(w.device) == route:
            route = None  # already there — routing would be a no-op copy
        chunk = int(os.environ.get("MASE_PHASE_BANK_CHUNK_ROWS", "4096"))
        if route is None and w.shape[0] <= chunk:
            return self._quantize_weight_with_config(w, cfg, block_dim)
        out = torch.empty_like(w)
        for i in range(0, w.shape[0], chunk):
            src = w[i : i + chunk]
            q = self._quantize_weight_with_config(
                src.to(route) if route else src, cfg, block_dim
            )
            out[i : i + chunk].copy_(q.to(w.device) if route else q)
        return out

    @torch.no_grad()
    def _build_phase_weight_banks(self) -> None:
        """Build per-phase weight banks from the FP weights in ``self.weight``.

        Must run while ``self.weight`` still holds the FP source (i.e. before
        the in-place prefill quantisation this method performs last).
        """

        device = self.weight.device
        self._decode_weight_q = torch.empty(0, device=device)
        self._decode_bias_q = torch.empty(0, device=device)
        self._decode_weight_fp = torch.empty(0, device=device)
        self._decode_bias_fp = torch.empty(0, device=device)

        prefill_quantises_weight = (
            not self.bypass and not self.gptq
            and self._weight_config_present(self.prefill_config)
        )

        if not self.shared_phase_banks:
            decode_cfg = self.decode_config
            if (
                not decode_cfg.get("bypass", False)
                and not self.decode_gptq
                and self._weight_config_present(decode_cfg)
            ):
                # Decode bank is quantised from the FP source, NOT from a
                # (possibly differently) quantised prefill bank.
                self._decode_weight_q = self._quantize_weight_banked(
                    self.weight.data, decode_cfg, block_dim=1
                )
                if self.bias is not None and self._bias_config_present(decode_cfg):
                    self._decode_bias_q = self._quantize_bias_with_config(
                        self.bias.data, decode_cfg
                    )
            elif decode_cfg.get("bypass", False) and prefill_quantises_weight:
                # Prefill-side flow: prefill quantises ``self.weight`` in
                # place below, so FP decode needs a pristine snapshot.
                self._decode_weight_fp = self.weight.data.detach().clone()
                if self.bias is not None:
                    self._decode_bias_fp = self.bias.data.detach().clone()

        if prefill_quantises_weight:
            self.weight.data.copy_(
                self._quantize_weight_banked(
                    self.weight.data, self.prefill_config, block_dim=1
                )
            )
            if self.bias is not None and self._bias_config_present(self.prefill_config):
                self.bias.data.copy_(
                    self._quantize_bias_with_config(self.bias.data, self.prefill_config)
                )

    @torch.no_grad()
    def adopt_decode_gptq_weight(self, weight: Tensor) -> None:
        """Install a GPTQ-calibrated decode weight bank.

        Called at module-replacement time when the GPTQ pre-pass ran with
        ``phase="decode"`` (GPTQ output stashed on the source ``nn.Linear``
        while ``weight`` itself was restored to FP for the prefill bank).
        """

        self._decode_weight_q = weight.detach().to(
            device=self.weight.device, dtype=self.weight.dtype, copy=True
        )

    @torch.no_grad()
    def adopt_decode_fp_snapshot(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Install an FP decode snapshot (prefill-side ``fp_only`` flow)."""

        self._decode_weight_fp = weight.detach().to(
            device=self.weight.device, dtype=self.weight.dtype, copy=True
        )
        if bias is not None and self.bias is not None:
            self._decode_bias_fp = bias.detach().to(
                device=self.bias.device, dtype=self.bias.dtype, copy=True
            )

    @torch.no_grad()
    def collapse_to_decode_bank(self) -> None:
        """Fold the decode bank into ``self.weight`` and drop the FP copy.

        Halves resident memory when only decode-phase numerics will ever be
        scored (e.g. decode-perplexity-only evaluation). After this, prefill
        forwards see the decode-quantised weights too, so callers that need a
        faithful FP prefill (task generation, prefill scoring) must NOT call
        this. No-op when there is no quantised decode bank.
        """
        if self._decode_weight_q.numel() == 0:
            return
        self.weight.data.copy_(self._decode_weight_q.to(self.weight.dtype))
        self._decode_weight_q = torch.empty(0, device=self.weight.device)
        if self.bias is not None and self._decode_bias_q.numel() > 0:
            self.bias.data.copy_(self._decode_bias_q.to(self.bias.dtype))
        self._decode_bias_q = torch.empty(0, device=self.weight.device)

    def _select_decode_bank(self) -> tuple[Tensor, Tensor | None]:
        if self._decode_weight_q.numel() > 0:
            bias = (
                self._decode_bias_q
                if self.bias is not None and self._decode_bias_q.numel() > 0
                else self.bias
            )
            return self._decode_weight_q, bias
        if self._decode_weight_fp.numel() > 0:
            bias = (
                self._decode_bias_fp
                if self.bias is not None and self._decode_bias_fp.numel() > 0
                else self.bias
            )
            return self._decode_weight_fp, bias
        return self.weight, self.bias

    # ---- construction seams -----------------------------------------------
    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: dict):
        """Create a phase-aware MX linear that REUSES the original Parameters.

        Unlike ``__init__`` which allocates fresh weights, this shares the
        original module's ``weight`` / ``bias`` Parameters directly. This is
        critical for tensor-parallel models (DTensor shards survive) and for
        the rotation-search swap path (plain <-> rotate swaps stay zero-copy).

        When ``linear`` is itself a phase-aware MX linear (rotation-search
        toggling), its banks are adopted as-is and nothing is requantised.
        When ``linear`` is a plain ``nn.Linear`` holding FP source weights,
        banks are built exactly as in ``load_state_dict``, including pickup
        of a GPTQ decode stash left by ``run_gptq(phase="decode")``.
        """
        assert config is not None, "config is None!"
        new = cls.__new__(cls)
        torch.nn.Module.__init__(new)

        # nn.Linear attributes
        new.in_features = linear.in_features
        new.out_features = linear.out_features
        new.weight = linear.weight  # share Parameter (may be DTensor)
        new.bias = linear.bias  # share Parameter (may be DTensor)

        # _LinearBase attributes
        new.pruning_masks = None

        new._init_phase_state(config)
        new._register_phase_bank_buffers()

        if isinstance(linear, _PhaseAwareMXLinearBase):
            # Zero-copy class swap: weights/banks already quantised — adopt.
            new._decode_weight_q = linear._decode_weight_q
            new._decode_bias_q = linear._decode_bias_q
            new._decode_weight_fp = linear._decode_weight_fp
            new._decode_bias_fp = linear._decode_bias_fp
            return new

        new._build_phase_weight_banks()
        gptq_stash = getattr(linear, GPTQ_DECODE_WEIGHT_ATTR, None)
        if gptq_stash is not None:
            new.adopt_decode_gptq_weight(gptq_stash)
        return new

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load pretrained (FP source) weights, then build per-phase banks."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._build_phase_weight_banks()
        return result

    # ---- runtime ------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x):
        runtime_phase = get_runtime_phase()
        if runtime_phase == "decode" and not self.shared_phase_banks:
            cfg = self.decode_config
            weight, bias = self._select_decode_bank()
        else:
            cfg = self.prefill_config
            weight, bias = self.weight, self.bias

        if cfg.get("bypass", False):
            return F.linear(x, weight, bias)

        x = self._quantize_activation_with_config(x, cfg)
        return F.linear(x, weight, bias)


class LinearMXFP(_PhaseAwareMXLinearBase):
    """MXFP linear with prefill/decode phase-aware execution."""

    _WEIGHT_CFG_KEYS = (
        "weight_block_size",
        "weight_exponent_width",
        "weight_frac_width",
    )
    _BIAS_CFG_KEYS = ("bias_block_size", "bias_exponent_width")

    def _quantize_weight_with_config(self, w: Tensor, cfg: dict, block_dim: int) -> Tensor:
        return mxfp_quantizer(
            w,
            block_size=cfg["weight_block_size"],
            element_exp_bits=cfg["weight_exponent_width"],
            element_frac_bits=cfg["weight_frac_width"],
            block_dim=block_dim,
        )

    def _quantize_bias_with_config(self, b: Tensor, cfg: dict) -> Tensor:
        return mxfp_quantizer(
            b,
            block_size=cfg["bias_block_size"],
            element_exp_bits=cfg["bias_exponent_width"],
            element_frac_bits=cfg.get("bias_frac_width"),
            block_dim=0,
        )

    # Activation quantisation: inherited key-dispatching base implementation
    # (supports MXINT and MXFP activations regardless of the weight format).


class LinearMXInt(_PhaseAwareMXLinearBase):
    """MXInt linear with prefill/decode phase-aware execution."""

    _WEIGHT_CFG_KEYS = ("weight_block_size", "weight_width")
    _BIAS_CFG_KEYS = ("bias_block_size", "bias_width")

    def _quantize_weight_with_config(self, w: Tensor, cfg: dict, block_dim: int) -> Tensor:
        return mxint_quantizer(
            w,
            block_size=cfg["weight_block_size"],
            element_bits=cfg["weight_width"],
            block_dim=block_dim,
            quantile_search=cfg.get("clip_search", False),
        )

    def _quantize_bias_with_config(self, b: Tensor, cfg: dict) -> Tensor:
        return mxint_quantizer(
            b,
            block_size=cfg["bias_block_size"],
            element_bits=cfg["bias_width"],
            block_dim=0,
            quantile_search=cfg.get("clip_search", False),
        )

    # Activation quantisation: inherited key-dispatching base implementation
    # (supports MXINT and MXFP activations regardless of the weight format).


class RotateMXFPLinear(LinearMXFP):
    """LinearMXFP + exact Hadamard rotation around the activation quantizer.

    Mirrors ``RotateMXIntLinear`` for MXFP. Reuses ``LinearMXFP`` for the
    weight-bank pipeline (``__init__`` / ``from_linear`` / ``load_state_dict``
    and phase dispatch); only the activation quantize swaps to
    ``mxfp_rotate_quantizer``. In a bypassed phase (e.g. FP prefill) the
    rotation round-trip is skipped entirely — it is a mathematical no-op in
    fp, so skipping preserves exact FP semantics.

    Extra config keys (optional):
        force_fp32_had: run the Hadamard multiplications in fp32.
    """

    def _quantize_activation_mxfp(self, x: Tensor, block_size: int, cfg: dict) -> Tensor:
        from chop.nn.quantizers.rotation import mxfp_rotate_quantizer

        return mxfp_rotate_quantizer(
            x,
            hadamard_dim=self.in_features,
            block_size=block_size,
            element_exp_bits=cfg["data_in_exponent_width"],
            element_frac_bits=cfg["data_in_frac_width"],
            block_dim=-1,
            quantile_search=cfg.get("clip_search", False),
            force_fp32=cfg.get("force_fp32_had", False),
        )

    def _quantize_activation_mxint(self, x: Tensor, block_size: int, cfg: dict) -> Tensor:
        from chop.nn.quantizers.rotation import mxint_rotate_quantizer

        return mxint_rotate_quantizer(
            x,
            hadamard_dim=self.in_features,
            block_size=block_size,
            element_bits=cfg["data_in_width"],
            block_dim=-1,
            quantile_search=cfg.get("clip_search", False),
            force_fp32=cfg.get("force_fp32_had", False),
        )


class RotateMXIntLinear(LinearMXInt):
    """LinearMXInt + exact Hadamard rotation around the activation quantizer.

    Identical to ``LinearMXInt`` for the weight-bank pipeline; the activation
    quantize is replaced with ``mxint_rotate_quantizer``: the input is rotated
    by an exact Hadamard, quantised, and rotated back. In a bypassed phase the
    round-trip is skipped (exact fp no-op), so FP prefill stays untouched.

    Extra config keys (optional):
        force_fp32_had: run the Hadamard multiplications in fp32.
    """

    def _quantize_activation_mxint(self, x: Tensor, block_size: int, cfg: dict) -> Tensor:
        from chop.nn.quantizers.rotation import mxint_rotate_quantizer

        return mxint_rotate_quantizer(
            x,
            hadamard_dim=self.in_features,
            block_size=block_size,
            element_bits=cfg["data_in_width"],
            block_dim=-1,
            quantile_search=cfg.get("clip_search", False),
            force_fp32=cfg.get("force_fp32_had", False),
        )
