
Implemented List:
- Wav2ec model, including:
    - new speech model type, 
    - 4 new packages in setup ("librosa", "soundfile", "jiwer", "pyctcdecode"), 
    - 3 new MASE_IMPLICIT_FUNCTIONS ("cumsum", "flip", "repeat"), 
    - 3 new MASE_MODULE_RELATED_FUNCS ("zeros", "setitem", "invert"), 
    - fixed common_metadata_layers "div" config parameters in func_data
    - added common_metadata_layers ("zeros", "setitem", "invert") w configs to func_data
    - added common_metadata_layers ("cumsum", "flip", "repeat") w configs to module_data
    
- DataCollatorCTCWithPadding, including: 
    - additional logic for defining a custom data collator in get_trainer()

- Added WER (Word Error Rate) as a new evaluate_metric within chop/tools/huggingface

- Added weight movement pruning, local and global, including:
    - added logic to get_weight_hook to pass correct info for global vs local cases
    - fixed l1-norm global pruning, also gave it a fall back low memory method in case of memory issues
    - also added fall back low memory method for global movement pruning

- Added optional parameters to get_trainer() for gradient accumalation and mini batch sizes
