#!/bin/bash
#SBATCH --job-name=cray_mpich_test
#SBATCH --output=%j_cray_mpich_test.out
#SBATCH --error=%j_cray_mpich_test.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --export=ALL

echo "=== Cray MPICH Python Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Load any necessary modules (adjust for your system)
# module load python/3.11
# module load mpi/cray-mpich

# Check MPI implementation
echo "=== MPI Implementation Info ==="
mpirun --version
echo ""

# Check Python MPI bindings
echo "=== Python MPI Bindings ==="
python -c "from mpi4py import MPI; print(f'MPI Library: {MPI.Get_library_version()}'); print(f'MPI Version: {MPI.Get_version()}')"
echo ""

# Run the Cray MPICH test
echo "=== Running Cray MPICH Test ==="
mpirun -np $SLURM_NTASKS python cray_mpich_test.py

echo ""
echo "=== Test completed ==="
