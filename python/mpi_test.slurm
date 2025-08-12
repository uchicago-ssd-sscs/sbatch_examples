#!/bin/bash
#SBATCH --job-name=MPI_Test
#SBATCH --output=%j_mpi_test.out
#SBATCH --error=%j_mpi_test.err
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=CPU
#SBATCH --nodes=2

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Activate conda environment
conda init
conda activate gpu

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
mpirun -np 2 --hostfile hostfile.txt python mpi_test.py

echo "MPI test completed"

bash slurm_diagnostic.sh
