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
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --distribution=block

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Activate conda environment from existing installation in home directory (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate openmpi || echo "Warning: could not activate conda env 'openmpi'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Configure MPI to use ONLY 10.150.x.x network (100G Ethernet)
echo "=== CONFIGURING MPI FOR 100G ETHERNET (10.150.x.x ONLY) ==="
export OMPI_MCA_btl_tcp_if_include="10.150.0.0/16"  # Only allow 10.150.x.x network
export OMPI_MCA_oob_tcp_if_include="10.150.0.0/16"  # Only allow 10.150.x.x network
export PRTE_MCA_oob_tcp_if_include="10.150.0.0/16"  # Only allow 10.150.x.x network

# Print SLURM environment
echo "=== OpenMPI Test (Exclusive Node Allocation) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Exclusive allocation: YES (no interference from other jobs)"
echo ""

# Test MPI
echo "Testing MPI multi-node communication..."

# Create hostfile for mpirun
echo "Creating hostfile..."
# Expand SLURM node list and create proper hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test OpenMPI functionality and performance
echo "=== RUNNING OPENMPI FUNCTIONALITY TEST ==="
echo "Testing OpenMPI features, performance, and collectives..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca oob_tcp_if_include "$OMPI_MCA_oob_tcp_if_include" \
  --mca btl_tcp_if_include "$OMPI_MCA_btl_tcp_if_include" \
  python mpi_test.py

echo "OpenMPI functionality test completed"

echo ""
echo "=== RUNNING OPENMPI NETWORK PERFORMANCE TEST ==="
echo "Testing OpenMPI network bandwidth, latency, and topology..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca oob_tcp_if_include "$OMPI_MCA_oob_tcp_if_include" \
  --mca btl_tcp_if_include "$OMPI_MCA_btl_tcp_if_include" \
  python network_bandwidth_test_v2.py

echo "OpenMPI network performance test completed"

bash slurm_diagnostic.sh
