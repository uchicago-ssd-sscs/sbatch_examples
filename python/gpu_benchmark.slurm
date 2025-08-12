#!/bin/bash
#SBATCH --job-name=GPU_Benchmark
#SBATCH --output=%j_gpu_benchmark.out
#SBATCH --error=%j_gpu_benchmark.err
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=H100

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load necessary modules (adjust based on your system)
# module load python/anaconda3
# module load cuda/11.8
# module load openmpi/4.1.4

# Activate a virtual environment (optional but recommended)
coda activate gpu

# Install required packages if not already installed (optional)
conda install -c conda-forge numpy mpi4py cupy-cuda11x torch psutil openmpi

# Run the GPU benchmark
echo "Starting GPU benchmark on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the benchmark
python gpu_benchmark.py

echo "GPU benchmark completed" 