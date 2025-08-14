#!/bin/bash
#SBATCH --job-name=quick_mpi_test
#SBATCH --output=%j_quick_test.out
#SBATCH --error=%j_quick_test.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --export=ALL

echo "=== Quick MPI Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo ""

# Load MPI module
module load hmpt/2.31

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate hpe_mpi || echo "Warning: could not activate conda env 'hpe_mpi'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Check MPI implementation
echo "=== MPI Implementation Info ==="
mpirun --version
echo ""

# Run quick test
echo "=== Running Quick MPI Test ==="
mpirun -np $SLURM_NTASKS python quick_mpi_test.py

echo ""
echo "=== Quick test completed ==="
