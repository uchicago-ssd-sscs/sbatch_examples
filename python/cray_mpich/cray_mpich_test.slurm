#!/bin/bash
#SBATCH --job-name=cray_mpich_test
#SBATCH --output=%j_cray_mpich_test.out
#SBATCH --error=%j_cray_mpich_test.err
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --hint=nomultithread
#SBATCH --distribution=block

echo "=== Cray MPICH Python Test (Exclusive Node Allocation) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Exclusive allocation: YES (no interference from other jobs)"
echo "Node list: $SLURM_NODELIST"
echo ""

# Load MPI modules (choose one based on your system)
# Option 1: HPE MPT (recommended for HPE systems)
# module load mpt/2.31

# Option 3: Alternative HPE module
module load hmpt/2.31

# Activate conda environment (non-interactive safe)
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

# Test MPI
echo "Testing MPI multi-node communication..."

# Create hostfile for mpirun
echo "Creating hostfile..."
# Expand SLURM node list and create proper hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test Cray MPICH functionality and performance
echo "=== RUNNING CRAY MPICH FUNCTIONALITY TEST ==="
echo "Testing Cray MPICH features, performance, and collectives..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  python mpi_test.py

echo "Cray MPICH functionality test completed"

echo ""
echo "=== RUNNING CRAY MPICH NETWORK PERFORMANCE TEST ==="
echo "Testing Cray MPICH network bandwidth, latency, and topology..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  python network_bandwidth_test_v2.py

echo "Cray MPICH network performance test completed"

echo ""
echo "=== Test completed ==="
