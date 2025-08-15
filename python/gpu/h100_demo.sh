#!/bin/bash

echo "🚀 Launching H100 GPU Demo Interactive Session"
echo "=============================================="
echo "This will start an interactive session on the H100 partition"
echo "and run the H100 demonstration script."
echo ""

# Check if we're already in a SLURM job
if [ -n "$SLURM_JOB_ID" ]; then
    echo "❌ Already running in a SLURM job. Please run this from a login node."
    exit 1
fi

echo "📋 Session Configuration:"
echo "   Partition: H100"
echo "   Tasks: 1"
echo "   CPUs per task: 16"
echo "   Memory: 128GB"
echo "   Time limit: 30 minutes"
echo ""

# Launch interactive session
echo "🎯 Starting interactive session..."
srun --partition=H100 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=00:30:00 \
     --pty bash -c "
echo '✅ Interactive session started on H100 partition'
echo '🚀 Setting up environment...'

# Load necessary modules (adjust based on your system)
# module load python/anaconda3
# module load cuda/11.8

# Activate conda environment from existing installation in home directory
if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then
    source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate gpu || echo 'Warning: could not activate conda env gpu'
else
    echo 'Warning: conda.sh not found; skipping conda activation'
fi

# Set environment variables for optimal H100 performance
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

# Show system information
echo ''
echo '📊 System Information:'
echo '   Hostname:' \$(hostname)
echo '   SLURM_JOB_ID:' \$SLURM_JOB_ID
echo '   CUDA_VISIBLE_DEVICES:' \$CUDA_VISIBLE_DEVICES
echo '   GPU Count:' \$SLURM_GPUS_ON_NODE
echo '   Memory:' \$SLURM_MEM_PER_NODE
echo '   CPUs:' \$SLURM_CPUS_ON_NODE
echo ''

# Show GPU information
echo '🔍 GPU Information:'
nvidia-smi

echo ''
echo '🚀 Running H100 Demo...'
echo ''

# Run the demonstration
python h100_demo.py

echo ''
echo '🎉 H100 demonstration completed!'
echo 'Interactive session will end in 30 seconds...'
sleep 30
"
