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

# Load MPI modules (choose one based on your system)
# Option 1: HPE MPT (recommended for HPE systems)
module load mpt/2.31

# Option 2: If using conda-installed MPI, uncomment these:
# conda activate your_mpi_env  # if you created a separate conda environment
# # No module loading needed for conda-installed MPI

# Option 3: Alternative HPE module
# module load hmpt/2.31

# Check MPI implementation
echo "=== MPI Implementation Info ==="
mpirun --version
echo ""

# Verify we're using the correct MPI implementation
echo "=== Verifying MPI Implementation ==="
mpirun --version
echo ""

# Check for specific MPI implementations
if mpirun --version 2>&1 | grep -i "hpe\|mpt" > /dev/null; then
    echo "✅ HPE MPT detected"
elif mpirun --version 2>&1 | grep -i "mpich" > /dev/null; then
    echo "✅ MPICH detected"
elif mpirun --version 2>&1 | grep -i "openmpi" > /dev/null; then
    echo "✅ OpenMPI detected"
else
    echo "⚠️  Unknown MPI implementation"
fi
echo ""

# Check Python MPI bindings
echo "=== Python MPI Bindings ==="
python -c "from mpi4py import MPI; print(f'MPI Library: {MPI.Get_library_version()}'); print(f'MPI Version: {MPI.Get_version()}')"
echo ""

# Run the Cray MPICH specific test
echo "=== Running Cray MPICH Specific Test ==="
mpirun -np $SLURM_NTASKS python cray_mpich_specific_test.py

echo ""
echo "=== Running Basic Cray MPICH Test ==="
mpirun -np $SLURM_NTASKS python cray_mpich_test.py

echo ""
echo "=== Test completed ==="
