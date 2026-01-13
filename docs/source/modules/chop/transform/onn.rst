chop.passes.module.transforms.onn
=================================

This module provides transformation passes for converting standard neural network
modules into Optical Neural Network (ONN) equivalents. The optical transformer
implementation is based on the `Optical Transformers paper <https://arxiv.org/abs/2302.10360>`_.

Optical neural networks leverage photonic hardware to perform matrix multiplications
with reduced power consumption. This transform simulates the quantization effects
and constraints of optical compute hardware, enabling model development and evaluation
before deployment on physical optical accelerators.

.. note::

   This module requires the ``mase-triton`` package to be installed.
   Install via: ``pip install mase-triton``


Transform Pass
--------------

optical\_transformer\_module\_transform\_pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: chop.passes.module.transforms.onn.optical_transformer_module_transform_pass


Configuration
-------------

The transform pass accepts configuration through the ``pass_args`` dictionary.
Layer matching can be done by exact name or regex patterns.

Example configuration:

.. code-block:: python

   pass_args = {
       "by": "regex_name",  # or "name" for exact matching
       "default": {
           "q_levels": 256,
           "q_lut_min": 0.020040,
           "q_smooth_factor": 0.9,
           "q_init_seed": 0,
           "q_bypass": False,
       },
       # Override for specific layers using regex
       ".*mlp.*": {
           "q_levels": 128,
           "q_bypass": False,
       },
   }


Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``q_levels``
     - int
     - 256
     - Number of quantization levels for optical simulation
   * - ``q_lut_min``
     - float
     - 0.020040
     - Minimum value for the lookup table used in quantization
   * - ``q_smooth_factor``
     - float
     - 0.9
     - Exponential moving average factor for updating running statistics
   * - ``q_init_seed``
     - int
     - 0
     - Random seed for quantization noise initialization
   * - ``q_bypass``
     - bool
     - False
     - If True, bypass optical quantization (useful for debugging)


Layers
------

OtLinear
^^^^^^^^

.. py:data:: chop.passes.module.transforms.onn.layers.linear.OtLinear

   Optical Transformer Linear layer.

   This is an alias to ``mase_triton.optical_compute.layers.OpticalTransformerLinear``.
   It replaces standard ``torch.nn.Linear`` layers with quantized optical transformer
   equivalents that simulate optical neural network hardware constraints.

   The layer applies quantization to both the input activations and weights during
   matrix multiplication, and tracks running min/max statistics for calibration.

   **Class method:**

   .. py:method:: from_linear(linear, **kwargs)
      :classmethod:

      Create an OtLinear from an existing ``torch.nn.Linear`` layer.

      :param linear: Source linear layer
      :type linear: torch.nn.Linear
      :param kwargs: Quantization parameters (q_levels, q_lut_min, etc.)
      :return: Optical transformer linear layer with copied weights


OtLlamaAttention
^^^^^^^^^^^^^^^^

.. autoclass:: chop.passes.module.transforms.onn.layers.attn.OtLlamaAttention
   :members:
   :undoc-members:
   :show-inheritance:


Functional API
--------------

optical\_transformer\_SDPA
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: chop.passes.module.transforms.onn.layers.attn.optical_transformer_SDPA


Usage Example
-------------

Basic usage with a LLaMA model:

.. code-block:: python

   from transformers import AutoModelForCausalLM
   from chop.passes.module.transforms.onn import optical_transformer_module_transform_pass

   # Load a pretrained model
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

   # Define transformation configuration
   pass_args = {
       "by": "regex_name",
       "default": {
           "q_levels": 256,
           "q_lut_min": 0.020040,
           "q_smooth_factor": 0.9,
           "q_init_seed": 0,
           "q_bypass": False,
       },
   }

   # Apply the optical transformer transform
   model = optical_transformer_module_transform_pass(model, pass_args)

   # The model now uses OtLinear and OtLlamaAttention layers
   # Continue with training or inference as usual


Selective Layer Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transform only specific layers using regex patterns:

.. code-block:: python

   pass_args = {
       "by": "regex_name",
       # Only transform attention layers
       ".*self_attn.*": {
           "q_levels": 256,
           "q_bypass": False,
       },
       # Transform MLP with different settings
       ".*mlp.*": {
           "q_levels": 128,
           "q_bypass": False,
       },
   }


Bypass Mode for Debugging
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``q_bypass=True`` to disable quantization while keeping the module structure:

.. code-block:: python

   pass_args = {
       "by": "regex_name",
       "default": {
           "q_levels": 256,
           "q_bypass": True,  # Disable quantization
       },
   }
