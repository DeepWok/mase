#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=32:mem=64gb:ngpus=1:gpu_type=RTX6000:cpu_type=rome
#PBS -N wav2vec_quick_test

module load anaconda3/personal
source /rds/general/user/at5424/home/anaconda3/etc/profile.d/conda.sh
conda activate mase_hpc

# Navigate to the repository directory
cd /rds/general/user/at5424/home/mase-individual

# Run the script with 0.1 epochs and 2 trials
python docs/custom_scripts_nyal/wav2vec_optimization/main.py --epochs 0.1 --n_trials 2