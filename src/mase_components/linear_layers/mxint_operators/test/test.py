import torch
from torch import Tensor
import torch.nn as nn
from functools import partial

# Import from utils directly
from utils import MXIntLinearHardware, MXIntMatmulHardware, mxint_softmax, mxint_quantize

class MxIntViTSelfAttentionHead(torch.nn.Module):
    def __init__(self, dim, attn_drop, q_config) -> None:
        super().__init__()
        self.dim = dim
        
        # Extract configs for different components
        self.linear_config = q_config["linear"]
        self.matmul_config = q_config["matmul"]
        self.softmax_config = q_config["softmax"]

        # Initialize hardware components with configs
        self.linear = MXIntLinearHardware
        self.matmul = MXIntMatmulHardware
        self.act = partial(mxint_softmax, q_config=self.softmax_config)

    def self_attention_head(
        self,
        mquery: torch.Tensor, equery: torch.Tensor,
        mkey: torch.Tensor, ekey: torch.Tensor,
        mvalue: torch.Tensor, evalue: torch.Tensor,
    ) -> Tensor:
        # Configure configs for matrix multiplications
        qk_x_config = {
            "width": self.matmul_config["A_MAN_WIDTH"],
            "exponent_width": self.matmul_config["A_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["A_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["A_COMPUTE_DIM1"],
            "depth_dim_0": mquery.shape[-1] // self.matmul_config["A_COMPUTE_DIM0"],
            "depth_dim_1": mquery.shape[-2] // self.matmul_config["A_COMPUTE_DIM1"],
            "dim_0": mquery.shape[-1],
            "dim_1": mquery.shape[-2],
        }
        
        qk_y_config = {
            "width": self.matmul_config["B_MAN_WIDTH"],
            "exponent_width": self.matmul_config["B_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["B_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["B_COMPUTE_DIM1"],
            "depth_dim_0": mkey.shape[-1] // self.matmul_config["B_COMPUTE_DIM0"],
            "depth_dim_1": mkey.shape[-2] // self.matmul_config["B_COMPUTE_DIM1"],
            "dim_0": mkey.shape[-1],
            "dim_1": mkey.shape[-2],
        }
        
        qk_out_config = {
            "width": self.matmul_config["OUT_MAN_WIDTH"],
            "exponent_width": self.matmul_config["OUT_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["C_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["C_COMPUTE_DIM1"],
            "depth_dim_0": mquery.shape[-2] // self.matmul_config["C_COMPUTE_DIM0"],
            "depth_dim_1": mkey.shape[-2] // self.matmul_config["C_COMPUTE_DIM1"],
            "dim_0": mquery.shape[-2],
            "dim_1": mkey.shape[-2],
        }

        print("\n=== Self Attention Head Debug Info ===")
        print("Query shape:", mquery.shape)
        print("Key shape:", mkey.shape)
        print("Value shape:", mvalue.shape)

        # First matmul: Q*K^T
        matt_scores, eatt_scores = self.matmul(
            mquery, equery,
            mkey.transpose(-1, -2), ekey,
            qk_x_config, qk_y_config, qk_out_config
        )

        print("\n--- Attention Scores ---")
        print("Shape:", matt_scores.shape)
        print("Sample values:", matt_scores[0, 0, :5])

        # Apply softmax
        mprobs, eprobs = self.act(matt_scores)
        
        print("\n--- Attention Probabilities ---")
        print("Shape:", mprobs.shape)
        print("Sample values:", mprobs[0, 0, :5])

        # Configure matmul for attn*V
        av_x_config = {
            "width": self.matmul_config["A_MAN_WIDTH"],
            "exponent_width": self.matmul_config["A_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["A_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["A_COMPUTE_DIM1"],
            "depth_dim_0": mprobs.shape[-1] // self.matmul_config["A_COMPUTE_DIM0"],
            "depth_dim_1": mprobs.shape[-2] // self.matmul_config["A_COMPUTE_DIM1"],
            "dim_0": mprobs.shape[-1],
            "dim_1": mprobs.shape[-2],
        }
        
        av_y_config = {
            "width": self.matmul_config["B_MAN_WIDTH"],
            "exponent_width": self.matmul_config["B_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["B_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["B_COMPUTE_DIM1"],
            "depth_dim_0": mvalue.shape[-1] // self.matmul_config["B_COMPUTE_DIM0"],
            "depth_dim_1": mvalue.shape[-2] // self.matmul_config["B_COMPUTE_DIM1"],
            "dim_0": mvalue.shape[-1],
            "dim_1": mvalue.shape[-2],
        }
        
        av_out_config = {
            "width": self.matmul_config["OUT_MAN_WIDTH"],
            "exponent_width": self.matmul_config["OUT_EXP_WIDTH"],
            "parallism_dim_0": self.matmul_config["C_COMPUTE_DIM0"],
            "parallism_dim_1": self.matmul_config["C_COMPUTE_DIM1"],
            "depth_dim_0": mvalue.shape[-1] // self.matmul_config["C_COMPUTE_DIM0"],
            "depth_dim_1": mprobs.shape[-2] // self.matmul_config["C_COMPUTE_DIM1"],
            "dim_0": mvalue.shape[-1],
            "dim_1": mprobs.shape[-2],
        }

        # Second matmul: attn*V
        mcontext, econtext = self.matmul(
            mprobs, eprobs,
            mvalue, evalue,
            av_x_config, av_y_config, av_out_config
        )
        
        print("\n--- Context Layer ---")
        print("Shape:", mcontext.shape)
        print("Sample values:", mcontext[0, 0, :5])
        print("===============================\n")
        
        # Reconstruct output
        return mcontext * (2 ** (econtext.unsqueeze(-1) - self.matmul_config["OUT_MAN_WIDTH"] + 1))

    def forward(
        self,
        mquery: torch.Tensor, equery: torch.Tensor,
        mkey: torch.Tensor, ekey: torch.Tensor,
        mvalue: torch.Tensor, evalue: torch.Tensor,
    ) -> Tensor:
        return self.self_attention_head(
            mquery, equery, mkey, ekey, mvalue, evalue
        )

def test_mxint_vit_self_attention():
    print("\n=== Starting MxIntViTSelfAttention Test ===")
    
    # Test parameters
    batch_size = 1
    num_heads = 8
    seq_length = 16
    head_dim = 64
    attn_drop = 0.1
    
    print("\nTest Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Head dimension: {head_dim}")
    
    # Create sample input tensors
    query = torch.randn(batch_size, seq_length, head_dim)
    key = torch.randn(batch_size, seq_length, head_dim)
    value = torch.randn(batch_size, seq_length, head_dim)
    
    print("\nInput Tensor Shapes:")
    print(f"Query: {query.shape}")
    print(f"Key: {key.shape}")
    print(f"Value: {value.shape}")

    # Comprehensive configuration for all components
    q_config = {
        "linear": {
            "data_in_width": 8,
            "data_in_exponent_width": 4,
            "data_in_parallelism": [2, 2],
            "weight_width": 8,
            "weight_exponent_width": 4,
            "weight_parallelism": [2, 2],
            "bias_width": 8,
            "bias_exponent_width": 4,
            "bias_parallelism": [2, 1],
            "data_out_width": 8,
            "data_out_exponent_width": 4,
            "data_out_parallelism": [2, 2],
        },
        "matmul": {
            "A_MAN_WIDTH": 8,
            "A_EXP_WIDTH": 4,
            "B_MAN_WIDTH": 8,
            "B_EXP_WIDTH": 4,
            "OUT_MAN_WIDTH": 8,
            "OUT_EXP_WIDTH": 4,
            "A_COMPUTE_DIM0": 2,
            "A_COMPUTE_DIM1": 2,
            "B_COMPUTE_DIM0": 2,
            "B_COMPUTE_DIM1": 2,
            "C_COMPUTE_DIM0": 2,
            "C_COMPUTE_DIM1": 2,
        },
        "softmax": {
            "in_man_width": 8,
            "in_exp_width": 4,
            "data_out_n_width": 4,
            "data_out_man_width": 8,
            "data_out_exp_width": 4,
        }
    }

    # Initialize the attention head
    attention = MxIntViTSelfAttentionHead(
        dim=head_dim,
        attn_drop=attn_drop,  # Remove num_heads as it's not used
        q_config=q_config
    )

    # Quantize inputs before passing to attention
    _, mquery, equery = mxint_quantize(
        query, 
        q_config["matmul"]["A_MAN_WIDTH"],
        q_config["matmul"]["A_EXP_WIDTH"]
    )
    _, mkey, ekey = mxint_quantize(
        key,
        q_config["matmul"]["B_MAN_WIDTH"],
        q_config["matmul"]["B_EXP_WIDTH"]
    )
    _, mvalue, evalue = mxint_quantize(
        value,
        q_config["matmul"]["B_MAN_WIDTH"],
        q_config["matmul"]["B_EXP_WIDTH"]
    )

    # Run forward pass with quantized inputs
    try:
        print("\nRunning forward pass...")
        output = attention(
            mquery, equery,
            mkey, ekey,
            mvalue, evalue
        )
        print("\nResults:")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {seq_length}, {head_dim})")
        print(f"Sample output values (first 5):")
        for i, val in enumerate(output[0,0,:5].tolist()):
            print(f"  [{i}]: {val:10.6f}")
        
        assert output.shape == (batch_size, seq_length, head_dim)
        print("\n✓ Test passed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {str(e)}")
        raise
    finally:
        print("\n=== Test Completed ===\n")

if __name__ == "__main__":
    test_mxint_vit_self_attention()