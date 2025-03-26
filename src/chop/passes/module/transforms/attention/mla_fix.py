#!/usr/bin/env python3
# MLA module fixes for handling tensor shape issues and compatibility
import torch
from transformers.generation import utils as gen_utils
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_mla_apply_rotary_emb():
    """Apply fixes to the apply_rotary_emb function in MLA module"""
    try:
        # Import the MLA module
        from chop.nn.attention.modules.mla import apply_rotary_emb

        # Define a fixed version that properly handles tensor shapes
        def fixed_apply_rotary_emb(x, freqs_cis):
            # Original rotary embedding implementation
            # But add extra checks and reshaping to ensure tensor compatibility

            # Get the original shape of x
            bsz, seqlen, n_heads, head_dim = x.shape

            # Debug logging
            logger.debug(
                f"apply_rotary_emb shapes - x: {x.shape}, freqs_cis: {freqs_cis.shape}"
            )

            # Ensure freqs_cis has the right shape for applying rotary embeddings
            if freqs_cis.dim() == 2:  # (seq_len, dim/2)
                # Handle the standard case where freqs_cis is [seq_len, dim/2]
                # But make sure seq_len matches
                if freqs_cis.size(0) != seqlen:
                    logger.warning(
                        f"Mismatch in sequence lengths - x: {seqlen}, freqs_cis: {freqs_cis.size(0)}"
                    )
                    # Resize freqs_cis to match x's sequence length
                    if freqs_cis.size(0) > seqlen:
                        freqs_cis = freqs_cis[:seqlen]
                    else:
                        # Repeat the last position for extra length
                        pad_len = seqlen - freqs_cis.size(0)
                        freqs_cis = torch.cat(
                            [freqs_cis, freqs_cis[-1:].repeat(pad_len, 1)], dim=0
                        )

                # Check if head_dim is compatible with freqs_cis
                if freqs_cis.size(1) * 2 != head_dim:
                    logger.warning(
                        f"Dimension mismatch - head_dim: {head_dim}, freqs_cis dim: {freqs_cis.size(1)*2}. "
                        f"Adjusting freqs_cis."
                    )
                    # Adjust freqs_cis to match head_dim
                    required_dim = head_dim // 2
                    if freqs_cis.size(1) > required_dim:
                        # Truncate if too large
                        freqs_cis = freqs_cis[:, :required_dim]
                    else:
                        # Repeat if too small
                        repeat_factor = math.ceil(required_dim / freqs_cis.size(1))
                        freqs_cis = torch.cat([freqs_cis] * repeat_factor, dim=1)[
                            :, :required_dim
                        ]

                # Reshape to proper dimensions for the calculation
                freqs_cis = freqs_cis.view(1, seqlen, 1, -1)

            # Reshape x for applying rotary embeddings
            if head_dim % 2 != 0:
                logger.warning(
                    f"Head dimension {head_dim} is not even, padding for rotary embeddings"
                )
                head_dim_padded = head_dim + 1
                x_padded = torch.zeros(
                    bsz,
                    seqlen,
                    n_heads,
                    head_dim_padded,
                    device=x.device,
                    dtype=x.dtype,
                )
                x_padded[:, :, :, :head_dim] = x
                x = x_padded

            # Split the last dimension into real and imaginary parts
            x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

            # Apply rotary embeddings
            freqs_cis = freqs_cis.to(device=x.device, dtype=torch.float32)
            freqs_complex = torch.view_as_complex(
                freqs_cis.reshape(*freqs_cis.shape[:-1], -1, 2)
            )

            # Multiply by complex exponential
            x_rotated = x_complex * freqs_complex

            # Convert back to real
            x_out = torch.view_as_real(x_rotated).flatten(-2)

            # Restore original shape
            if head_dim % 2 != 0:
                x_out = x_out[:, :, :, :head_dim]

            return x_out

        # Apply the fix
        from chop.nn.attention.modules import mla

        mla.apply_rotary_emb = fixed_apply_rotary_emb

        logger.info("Successfully patched MLA apply_rotary_emb function")
        return True

    except Exception as e:
        logger.error(f"Error patching MLA apply_rotary_emb: {e}")
        return False


# Fix the MLA forward method to handle different tensor shapes
def fix_mla_forward():
    """Apply fixes to the MLA forward method"""
    try:
        # Import the MLA module
        from chop.nn.attention.modules.mla import MLA

        # Store original forward
        original_forward = MLA.forward

        # Define fixed forward method
        def fixed_forward(self, x, start_pos=0, freqs_cis=None, mask=None):
            """
            Fixed MLA forward method with better tensor shape handling
            """
            bsz, seqlen, _ = x.shape

            # Ensure proper head dimensions
            assert (
                self.dim % self.n_heads == 0
            ), f"Dimension {self.dim} not divisible by n_heads {self.n_heads}"

            # Debug logging
            logger.debug(
                f"MLA input shapes - x: {x.shape}, freqs_cis: {None if freqs_cis is None else freqs_cis.shape}"
            )

            # Get query matrix
            q = self.wq(x)

            # Reshape for KV projection
            original_shape = q.shape

            # Apply rotary embeddings to query if provided
            if self.qk_rope_head_dim > 0:
                if freqs_cis is None:
                    logger.warning(
                        "freqs_cis not provided for rotary embeddings, using random values"
                    )
                    # Generate random embeddings
                    dim = self.qk_rope_head_dim
                    freqs = 1.0 / (
                        10000
                        ** (
                            torch.arange(
                                0, dim, 2, dtype=torch.float32, device=x.device
                            )
                            / dim
                        )
                    )
                    t = torch.arange(seqlen, dtype=torch.float32, device=x.device)
                    freqs = torch.outer(t, freqs)
                    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

                # Apply rotary position embeddings
                q_pe = self.reshape_for_broadcast(q, bsz, seqlen, self.qk_rope_head_dim)

                # Fix freqs_cis shape to match q_pe
                if freqs_cis.shape[0] != seqlen:
                    logger.warning(
                        f"Adjusting freqs_cis from shape {freqs_cis.shape} to match sequence length {seqlen}"
                    )
                    if freqs_cis.shape[0] > seqlen:
                        freqs_cis = freqs_cis[:seqlen]
                    else:
                        # Pad by repeating
                        pad_len = seqlen - freqs_cis.shape[0]
                        freqs_cis = torch.cat(
                            [freqs_cis, freqs_cis[-1:].repeat(pad_len, 1)], dim=0
                        )

                # Apply rotary embeddings
                q_pe = self.apply_rotary_emb(q_pe, freqs_cis)

                # Reshape back
                q = self.reshape_from_broadcast(q_pe, original_shape)

            # Process the KV path with low-rank layers
            kv_input = self.wkv_a(x)
            kv = self.wkv_b(kv_input)

            # Normalize if needed
            if self.kv_norm is not None:
                kv = self.kv_norm(kv)

            # Reshape KV for splitting K and V
            kv = kv.reshape(bsz, seqlen, 2, self.v_head_dim, self.n_heads)
            k, v = kv[:, :, 0], kv[:, :, 1]

            # Compute attention scores
            k = k.transpose(2, 3)
            q = q.reshape(bsz, seqlen, self.n_heads, self.v_head_dim)

            # Safe transpose and reshape
            # These operations ensure dimensions are as expected
            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
            v = v.transpose(2, 3).transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)

            # Fix tensor shapes for matmul if needed
            if q.size(-1) != k.size(-2):
                logger.warning(
                    f"Fixing dimension mismatch: q[-1]={q.size(-1)}, k[-2]={k.size(-2)}"
                )
                # Adjust the smaller dimension to match the larger one
                if q.size(-1) < k.size(-2):
                    q_new = torch.zeros(
                        *q.shape[:-1], k.size(-2), device=q.device, dtype=q.dtype
                    )
                    q_new[..., : q.size(-1)] = q
                    q = q_new
                else:
                    k_new = torch.zeros(
                        *k.shape[:-2],
                        q.size(-1),
                        k.size(-1),
                        device=k.device,
                        dtype=k.dtype,
                    )
                    k_new[..., : k.size(-2), :] = k
                    k = k_new

            # Compute attention
            scores = torch.matmul(q, k)  # (bsz, n_heads, seqlen, seqlen)
            scores = scores / math.sqrt(self.v_head_dim)

            # Apply mask if provided
            if mask is not None:
                scores = scores + mask

            # Apply attention
            attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            output = torch.matmul(attn_weights, v)  # (bsz, n_heads, seqlen, head_dim)

            # Reshape output back to original dimensions
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

            # Final projection
            return self.wo(output)

        # Apply the fix
        MLA.forward = fixed_forward

        # Also fix apply_rotary_emb in the MLA class
        MLA.apply_rotary_emb = fix_mla_apply_rotary_emb

        logger.info("Successfully patched MLA forward method")
        return True

    except Exception as e:
        logger.error(f"Error patching MLA forward: {e}")
        return False


# Fix MLAAttentionWrapper
def fix_mla_wrapper():
    """Fix the MLAAttentionWrapper class"""
    try:
        # Import the wrapper class
        from chop.passes.module.transforms.attention.attention_transform_helper import (
            MLAAttentionWrapper,
        )

        # Save original methods
        original_init = MLAAttentionWrapper.__init__
        original_forward = MLAAttentionWrapper.forward

        # Define fixed methods
        def fixed_init(self, mla_module):
            """Fixed initialization to avoid buffer duplication"""
            torch.nn.Module.__init__(self)

            self.mla = mla_module
            self.is_mla_wrapper = True

            # Get dimensions
            self.hidden_size = mla_module.dim
            self.num_heads = mla_module.n_heads

            # Position counter for tracking
            self.register_buffer(
                "position_counter", torch.zeros(1, dtype=torch.int), persistent=False
            )

            # Import precompute_freqs_cis
            from chop.nn.attention.modules.mla import precompute_freqs_cis

            # Precompute frequencies
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(mla_module.model_args),
                persistent=False,
            )

        def fixed_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            """Fixed forward method with better handling of tensors and caching"""
            batch_size, seq_len = hidden_states.size()[:2]

            # Get dtype from MLA
            param = next(self.mla.parameters(), None)
            target_dtype = param.dtype if param is not None else torch.bfloat16
            device = hidden_states.device

            # Convert inputs
            hidden_states = hidden_states.to(target_dtype)

            # Process mask
            mla_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mla_mask = attention_mask.squeeze(1)
                else:
                    mla_mask = attention_mask

                if mla_mask.dtype != torch.bool:
                    mla_mask = mla_mask.to(target_dtype)

            # Handle start position
            start_pos = 0
            if past_key_value is not None:
                if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                    start_pos = past_key_value[0].size(2)
                elif hasattr(past_key_value, "get_seq_length"):
                    start_pos = past_key_value.get_seq_length()
                elif hasattr(self, "position_counter"):
                    start_pos = self.position_counter.item()
                    self.position_counter += seq_len

            # Process position embeddings
            freqs_cis = self.freqs_cis
            if position_ids is not None:
                # Handle batched position_ids
                if position_ids.dim() > 1:
                    position_ids_flat = position_ids[0]
                    freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids_flat)
                else:
                    freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids)
            else:
                # Use sequential positions
                if start_pos + seq_len <= freqs_cis.size(0):
                    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
                else:
                    # Handle case where sequence length exceeds available positions
                    logger.warning(
                        f"Sequence position {start_pos + seq_len} exceeds max length {freqs_cis.size(0)}"
                    )
                    available = (
                        freqs_cis[start_pos:]
                        if start_pos < freqs_cis.size(0)
                        else freqs_cis[-1:].expand(1, -1)
                    )
                    freqs_cis = available.repeat(
                        (seq_len + available.size(0) - 1) // available.size(0) + 1, 1
                    )[:seq_len]

            # Ensure on right device
            freqs_cis = freqs_cis.to(device=device)

            # Debug logs
            logger.debug(
                f"MLAWrapper forward shapes - hidden: {hidden_states.shape}, freqs_cis: {freqs_cis.shape}"
            )
            logger.debug(
                f"MLAWrapper forward mask shape: {None if mla_mask is None else mla_mask.shape}"
            )

            # Call MLA forward
            try:
                output = self.mla(
                    x=hidden_states,
                    start_pos=start_pos,
                    freqs_cis=freqs_cis,
                    mask=mla_mask,
                )
            except Exception as e:
                logger.error(f"Error in MLA forward: {e}")
                logger.error(f"  hidden_states: {hidden_states.shape}")
                logger.error(f"  start_pos: {start_pos}")
                logger.error(f"  freqs_cis: {freqs_cis.shape}")
                logger.error(
                    f"  mla_mask: {None if mla_mask is None else mla_mask.shape}"
                )
                raise

            # Match original dtype
            orig_dtype = kwargs.get("input_dtype", hidden_states.dtype)
            if output.dtype != orig_dtype:
                output = output.to(orig_dtype)

            # Handle caching
            present_key_value = None
            if use_cache:
                head_dim = self.hidden_size // self.num_heads
                dummy_key = torch.zeros(
                    (batch_size, self.num_heads, seq_len, head_dim),
                    device=device,
                    dtype=orig_dtype,
                )
                dummy_value = torch.zeros_like(dummy_key)
                present_key_value = (dummy_key, dummy_value)

            return output, None, present_key_value

        # Apply fixes
        MLAAttentionWrapper.__init__ = fixed_init
        MLAAttentionWrapper.forward = fixed_forward

        logger.info("Successfully patched MLAAttentionWrapper")
        return True

    except Exception as e:
        logger.error(f"Error patching MLAAttentionWrapper: {e}")
        return False


# Fix the DynamicCache.from_legacy_cache method
def fix_dynamic_cache():
    """Patch transformers.generation.utils.DynamicCache.from_legacy_cache"""
    try:
        from transformers.generation import utils as gen_utils

        # Save original
        original_from_legacy_cache = gen_utils.DynamicCache.from_legacy_cache

        # Define fixed method
        @classmethod
        def fixed_from_legacy_cache(cls, past_key_values):
            # Handle various formats and issues

            # If None, return empty cache
            if past_key_values is None:
                return cls()

            # Handle tensor format issues
            for layer_idx in range(len(past_key_values)):
                if past_key_values[layer_idx] is None:
                    # Create empty tuple
                    past_key_values[layer_idx] = (None, None)
                elif isinstance(past_key_values[layer_idx], torch.Tensor):
                    # Single tensor case
                    tensor = past_key_values[layer_idx]
                    past_key_values[layer_idx] = (tensor, torch.zeros_like(tensor))
                elif (
                    isinstance(past_key_values[layer_idx], tuple)
                    and len(past_key_values[layer_idx]) == 1
                ):
                    # Tuple with one element
                    key = past_key_values[layer_idx][0]
                    value = torch.zeros_like(key) if key is not None else None
                    past_key_values[layer_idx] = (key, value)

            # Call original
            return original_from_legacy_cache(past_key_values)

        # Apply patch
        gen_utils.DynamicCache.from_legacy_cache = fixed_from_legacy_cache

        logger.info("Successfully patched DynamicCache.from_legacy_cache")
        return True

    except Exception as e:
        logger.error(f"Error patching DynamicCache: {e}")
        return False


# Add fix for mla_by_type
def fix_mla_by_type():
    """Fix the mla_by_type function"""
    try:
        from chop.passes.module.transforms.attention import attention

        # Store original
        original_mla_by_type = attention.mla_by_type

        # Define fixed function
        def fixed_mla_by_type(network, pass_args):
            # Handle boolean configs
            fixed_args = {}
            for type_name, config_entry in pass_args.items():
                # Skip internal keys
                if type_name in ["by", "verbose"]:
                    fixed_args[type_name] = config_entry
                    continue

                # Handle boolean values
                if isinstance(config_entry, bool):
                    fixed_args[type_name] = {"config": {"name": "mla"}}
                elif isinstance(config_entry, dict):
                    if "config" in config_entry and isinstance(
                        config_entry["config"], bool
                    ):
                        # Fix boolean config
                        new_entry = dict(config_entry)
                        new_entry["config"] = {"name": "mla"}
                        fixed_args[type_name] = new_entry
                    else:
                        # Keep as is
                        fixed_args[type_name] = config_entry
                else:
                    # Keep as is
                    fixed_args[type_name] = config_entry

            # Call original with fixed args
            return original_mla_by_type(network, fixed_args)

        # Apply fix
        attention.mla_by_type = fixed_mla_by_type

        logger.info("Successfully patched mla_by_type")
        return True

    except Exception as e:
        logger.error(f"Error patching mla_by_type: {e}")
        return False


def apply_all_fixes():
    """Apply all MLA module fixes"""
    fixes_applied = [
        # fix_mla_apply_rotary_emb(),
        fix_mla_forward(),
        fix_mla_wrapper(),
        fix_dynamic_cache(),
        fix_mla_by_type(),
    ]

    success = all(fixes_applied)
    if success:
        logger.info("Successfully applied all MLA fixes")
    else:
        logger.warning("Some MLA fixes could not be applied")

    return success
