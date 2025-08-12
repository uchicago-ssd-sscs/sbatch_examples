#!/bin/bash
#SBATCH --job-name=MPI_Test
#SBATCH --output=%j_mpi_test.out
#SBATCH --error=%j_mpi_test.err
#SBATCH --export=ALL
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=CPU
#SBATCH --nodes=5

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Activate conda environment
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
echo $SLURM_JOB_NODELIST | tr ',' '\n' > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Try different MPI launchers
echo "Attempting MPI launch with different configurations..."

# Method 1: Use mpirun with hostfile
echo "Method 1: Using mpirun with hostfile..."
mpirun -np 2 --hostfile hostfile.txt python mpi_test.py

# If that fails, try Method 2
if [ $? -ne 0 ]; then
    echo "Method 1 failed, trying Method 2: Using srun with openmpi..."
    srun --mpi=openmpi python mpi_test.py
fi

# If that fails, try Method 3
if [ $? -ne 0 ]; then
    echo "Method 2 failed, trying Method 3: Using srun without MPI flag..."
    srun python mpi_test.py
fi

echo "MPI test completed"
