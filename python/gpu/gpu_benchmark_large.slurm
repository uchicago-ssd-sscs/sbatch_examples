#!/bin/bash
#SBATCH --job-name=Large_GPU_Benchmark
#SBATCH --output=%j_large_gpu_benchmark.out
#SBATCH --error=%j_large_gpu_benchmark.err
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=H100

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

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the large-scale GPU benchmark
echo "Starting Large-Scale GPU benchmark on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the large-scale benchmark
python gpu_benchmark_large.py

echo "Large-scale GPU benchmark completed"
