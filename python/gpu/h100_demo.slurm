#!/bin/bash
#SBATCH --job-name=H100_Demo
#SBATCH --output=%j_h100_demo.out
#SBATCH --error=%j_h100_demo.err
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --partition=H100
#SBATCH --gres=gpu:2

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load necessary modules (adjust based on your system)
# module load python/anaconda3
# module load cuda/11.8

# Activate conda environment from existing installation in home directory (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate gpu || echo "Warning: could not activate conda env 'gpu'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Set environment variables for optimal H100 performance
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

# Install required packages if needed
# pip install pynvml

# Run the H100 demonstration
echo "🚀 Starting H100 GPU Power Demonstration"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Count: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "CPUs: $SLURM_CPUS_ON_NODE"

# Show GPU information
nvidia-smi

# Run the demonstration
python h100_demo.py

echo "🎉 H100 demonstration completed!"
