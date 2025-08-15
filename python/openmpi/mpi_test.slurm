#!/bin/bash
#SBATCH --job-name=MPI_Test
#SBATCH --output=%j_mpi_test.out
#SBATCH --error=%j_mpi_test.err
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=CPU
#SBATCH --nodes=2

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Activate conda environment from existing installation in home directory (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate openmpi || echo "Warning: could not activate conda env 'openmpi'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Print SLURM environment
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Test MPI
echo "Testing MPI multi-node communication..."

# Create hostfile for mpirun
echo "Creating hostfile..."
# Expand SLURM node list and create proper hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test MPI with proper hostfile
echo "Testing MPI with expanded hostfile..."
echo "Launching 10 MPI processes across 5 nodes..."
mpirun -np 10 --hostfile hostfile.txt --bind-to none python mpi_test.py

echo "MPI test completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST ==="
echo "Testing actual cross-node network performance..."
mpirun -np 10 --hostfile hostfile.txt --bind-to none python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

bash slurm_diagnostic.sh
