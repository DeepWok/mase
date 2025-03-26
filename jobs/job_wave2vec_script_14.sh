#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=32:mem=64gb:ngpus=1:gpu_type=RTX6000:cpu_type=rome
#PBS -N wave2vec_quantise

module load anaconda3/personal
source /rds/general/user/at5424/home/anaconda3/etc/profile.d/conda.sh
conda activate mase_hpc

# Navigate to the repository directory
cd /rds/general/user/at5424/home/mase-individual

# Run the updated script
python docs/custom_scripts_tony/14_wave2vec_quantise_onnx.py
