import torch
from torch import Tensor
from .utils import my_clamp, my_round


def programming_noise(weight, gmax):
    """
    Implements PCM programming noise model:
    g_prog = g_T + N(0, σ_prog)
    σ_prog = max(-1.1731g_T^2 + 1.9650g_T + 0.2635, 0)
    
    Args:
        weight (torch.Tensor): Target weight values (g_T)
    
    Returns:
        torch.Tensor: Noisy weight values (g_prog)
    """
    # Calculate σ_prog using the quadratic equation
    sigma_prog = -1.1731 * (weight/gmax)**2 + 1.9650 * (weight/gmax) + 0.2635

    # Ensure σ_prog is non-negative
    # sigma_prog_shape = sigma_prog.shape
    # sigma_prog = sigma_prog.reshape(*sigma_prog_shape[:-2], -1)
    # sigma_prog = sigma_prog.max(dim=-1, keepdim=True).values + 1e-9
    # sigma_prog = sigma_prog.unsqueeze(-1)
    
    # Add noise from normal distribution N(0, σ_prog)
    noise = torch.randn_like(weight) * sigma_prog
    g_prog = weight + noise
    
    return g_prog

def get_read_noise(analog_weight, analog_x, result):
    """
    Implements short-term PCM read noise model:
    Calculate weight-dependent noise standard deviation
    σ_i^W = σ_0^W * sqrt(sum_j |w_ij| |x_j|^2)
    σ_0^W = 0.0175
    Args:
        analog_weight (torch.Tensor): Weight values
        analog_x (torch.Tensor): Input values
        result (torch.Tensor): Result of analog matrix multiplication
    Returns:
        torch.Tensor: Result with read noise
    """
    sigma_0 = 0.0175
    
    # Calculate the weight-dependent term
    # Using absolute values of weights and squared inputs
    noise_term = torch.sqrt((analog_x ** 2) @ torch.abs(analog_weight))
    
    # Generate noise with the calculated standard deviation
    noise = torch.randn_like(result) * sigma_0 * noise_term
    
    return noise

def get_ir_drop(analog_weight, analog_x, config):
    """
    Implements IR drop noise model:
    Δy_i^IR-drop = 0.0001 * (Σ_j |w_ij| |x_j|^2)
    """
    gamma = 0.35 * 5 * 1e-6
    n = analog_x.shape[-1]
    a = gamma * n * ((analog_x.abs()) @ (analog_weight.abs()))
    c = 0.05 * (a**3) - 0.2 *(a**2) + 0.5 * a
    j = torch.arange(analog_x.shape[-1], device=analog_x.device)
    ir_drop_scale = 1 - (1 - (j / n) ) ** 2

    ir_drop = - c * (analog_x * ir_drop_scale @ analog_weight)

    return ir_drop

def pcm_mm_core(analog_x, analog_weight, config):
    """
    Implements the core analog matrix multiplication with noise components.
    
    The equation for PCM-based computation can be represented as:
    y_i = σ^out·ξ_i + (Δy_i^IR-drop + Σ_j((w_ij + σ^W·ξ_ij)·x_j)
    
    Where:
    - σ^out·ξ_i: Read noise component
    - Δy_i^IR-drop: IR drop noise
    - σ^W·ξ_ij: Weight noise component
    - x_j: Input values
    
    Args:
        analog_x (torch.Tensor): Quantized input tensor
        analog_weight (torch.Tensor): Quantized weight tensor
        config (dict): Configuration parameters for noise simulation
        
    Returns:
        torch.Tensor: Result of analog matrix multiplication with noise effects
    """
    # Notice the normalized weight here is just the normalized conductance
    # First we need to transform it to real conductance
    # Assume the gmax is 5us(the result is from the original paper)
    if config.get("programming_noise", False):
        analog_weight = programming_noise(analog_weight, config.get("gmax", 5))
    else:
        analog_weight = analog_weight
    
    result = analog_x @ analog_weight

    if config.get("read_noise", False):
        read_noise = get_read_noise(analog_weight, analog_x, result)
    else:
        read_noise = 0

    if config.get("ir_drop", False):
        ir_drop = get_ir_drop(analog_weight, analog_x, config)
    else:
        ir_drop = 0


    if config.get("out_noise", False):
        out_noise = 0.04
    else:
        out_noise = 0

    result = result + ir_drop + read_noise + out_noise * torch.randn_like(result)

    return result

def dac_simulation(
    x: Tensor, width: int
):
    """
    y_i = α·γ_i·quant_out(F_i(quant_in(x/α)))
    """
    x_shape = x.shape
    x = x.reshape(*x_shape[:-2], -1)
    x_max = x.abs().max(dim=-1, keepdim=True).values + 1e-9
    
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    
    scale = 2**(width - 1) / x_max

    data_int = my_clamp(my_round(x.mul(scale)), int_min, int_max)
    data_q = data_int.div(scale)

    data_int = data_int / 2**(width - 1)
    data_scale = scale / 2**(width - 1)

    data_q = data_q.reshape(*x_shape)
    data_int = data_int.reshape(*x_shape)
    data_scale = data_scale.unsqueeze(-1)

    return data_q, data_int, data_scale

def adc_simulation(
    x: Tensor, width: int = 8, output_bound: float = 10.0
):
    bounded_x = my_clamp(x, -output_bound, output_bound)

    # support bias already
    x_shape = x.shape
    bounded_x_max = bounded_x.max(dim=-1, keepdim=True).values + 1e-9
    bounded_x_min = bounded_x.min(dim=-1, keepdim=True).values + 1e-9

    bias = (bounded_x_max + bounded_x_min) / 2

    biased_x = bounded_x - bias
    biased_x_max = biased_x.max(dim=-1, keepdim=True).values

    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
    
    scale = 2**(width - 1) / biased_x_max

    data_int = my_clamp(my_round(biased_x.mul(scale)), int_min, int_max)
    data_q = data_int.div(scale) + bias

    data_int = data_int / 2**(width - 1)
    data_scale = scale / 2**(width - 1)

    data_q = data_q.reshape(*x_shape)
    data_int = data_int.reshape(*x_shape)
    data_scale = data_scale.unsqueeze(-1)

    return data_q, data_int, data_scale

def pcm_tile(x, weight, config):
    """
    Implements noisy matrix multiplication for PCM-based computation.

    The general equation for PCM-based computation can be represented as:
    y_i = α·γ_i·quant_out(F_i(quant_in(x/α)))
    
    Where:
    - quant_in: Input quantization for the DAC
    - F_i: Real Computation with noise (programming noise, read noise, etc.)
    - quant_out: Output quantization for ADC
    - α, γ: Scaling factors for x and weight
    - β: Bias term
    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Weight tensor
        bias (torch.Tensor): Bias tensor
        config (dict): Configuration parameters for noise and quantization
        
    Returns:
        torch.Tensor: Result of noisy matrix multiplication
    """
    width = config.get("width", 8)
    is_signed = config.get("is_signed", True)
    gmax = config.get("gmax", 5)
    
    x_quant, analog_x, scale_x = dac_simulation(x, width)
    analog_weight, analog_weight_scale = weight_normalize(weight, gmax, config)

    analog_out = pcm_mm_core(analog_x, analog_weight, config)
    # analog_out = analog_x @ analog_weight

    adc_out, _, _ = adc_simulation(analog_out, width=8, output_bound=10.0 * gmax)

    result = adc_out.div(scale_x).div(analog_weight_scale)

    return result

def weight_normalize(weight, gmax, config):
    weight_shape = weight.shape
    weight_array = weight.reshape(-1, config.get("core_size", 256)**2)
    weight_max = weight_array.abs().max(dim=-1, keepdim=True).values + 1e-9
    weight_scale = gmax / weight_max
    analog_weight_array = weight_array * weight_scale
    analog_weight = analog_weight_array.reshape(*weight_shape)
    analog_weight_scale = weight_scale.reshape(*weight_shape[:-2], 1,1)
    return analog_weight, analog_weight_scale