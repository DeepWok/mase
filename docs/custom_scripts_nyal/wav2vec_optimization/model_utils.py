"""
MASE graph setup and model construction.
"""

import torch
import logging
from chop import MaseGraph
import chop.passes as passes
from chop.models import CombinedWav2Vec2CTC

# Set up logging
logger = logging.getLogger(__name__)

def setup_mase_graph(encoder):
    """Create and initialize MASE graph with metadata"""
    logger.info("Setting up MASE graph...")
    
    # Create MASE graph
    mg = MaseGraph(
        encoder,
        hf_input_names=["input_values", "attention_mask"],
    )
    
    # Initialize metadata
    mg, _ = passes.init_metadata_analysis_pass(mg)
    
    # Create dummy input for analysis
    dummy_in = {
        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long),
    }
    
    # Add common metadata
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": dummy_in,
            "add_value": True,
            "force_device_meta": False,
        }
    )
    
    logger.info("MASE graph setup complete")
    
    return mg, dummy_in

def create_combined_model(encoder, ctc_head, decoder):
    """Create a combined model with encoder, CTC head, and decoder"""
    logger.info("Creating combined model...")
    
    combined_model = CombinedWav2Vec2CTC(
        encoder=encoder,
        ctc_head=ctc_head,
        decoder=decoder,
        beam_width=10
    )
    
    logger.info("Combined model created successfully")
    
    return combined_model
