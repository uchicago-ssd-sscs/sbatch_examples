#!/bin/bash
#SBATCH --job-name=MPI_Test_Force_RoCE
#SBATCH --output=%j_mpi_test_force_roce.out
#SBATCH --error=%j_mpi_test_force_roce.err
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=CPU
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --distribution=block

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Activate conda environment (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate openmpi || echo "Warning: could not activate conda env 'openmpi'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Force UCX/RoCE configuration for maximum performance
echo "=== FORCING UCX/RoCE CONFIGURATION ==="

# Method 1: Force UCX transport (preferred for RoCE)
export OMPI_MCA_pml=ucx              # Use UCX Point-to-Point Messaging Layer
export OMPI_MCA_osc=ucx              # Use UCX One-Sided Communications
export OMPI_MCA_spml=ucx             # Use UCX Shared Memory

# Method 2: UCX transport layer selection
export UCX_TLS=rc,ud,dc,shm          # InfiniBand transports + shared memory
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1  # Try multiple Mellanox devices
# Alternative if above fails:
# export UCX_TLS=tcp,shm             # Fallback to TCP if no RDMA

# Method 3: Force specific device types (try multiple)
export UCX_IB_DEVICE_SPECS=mlx5_0:roce_v2,mlx5_1:roce_v2  # Force RoCE v2
# Alternative device detection:
# export UCX_NET_DEVICES=bond1        # Use specific interface
# export UCX_NET_DEVICES=ib0,ib1      # Use InfiniBand if available

# Method 4: Optimize UCX for performance
export UCX_RC_MLX5_TM_ENABLE=n       # Disable tag matching (can be slow)
export UCX_IB_REG_METHODS=rcache     # Use registration cache
export UCX_MM_SEG_SIZE=8192          # Shared memory segment size

# Method 5: Disable conflicting BTLs to force UCX path
export OMPI_MCA_btl=^vader,^tcp,^openib,^uct  # Disable competing transports

# Print configuration
echo "=== UCX/RoCE CONFIGURATION ==="
echo "OMPI_MCA_pml: $OMPI_MCA_pml"
echo "OMPI_MCA_osc: $OMPI_MCA_osc"
echo "UCX_TLS: $UCX_TLS"
echo "UCX_NET_DEVICES: $UCX_NET_DEVICES"
echo "UCX_IB_DEVICE_SPECS: $UCX_IB_DEVICE_SPECS"

# Print SLURM environment
echo "=== OpenMPI Test (Forced RoCE/UCX) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Exclusive allocation: YES (no interference from other jobs)"
echo ""

# Check available RDMA devices
echo "=== RDMA DEVICE DETECTION ==="
echo "Checking for RDMA devices..."
ibv_devinfo 2>/dev/null | head -20 || echo "No ibv_devinfo available"

echo "Checking UCX devices..."
ucx_info -d 2>/dev/null | head -10 || echo "No ucx_info available"

echo "Checking network interfaces..."
ip link show | grep -E "(bond|ib|mlx)" || echo "No special network interfaces found"

# Test MPI
echo "Testing MPI multi-node communication with forced RoCE/UCX..."

# Create hostfile for mpirun
echo "Creating hostfile..."
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test OpenMPI functionality with verbose UCX output
echo "=== RUNNING OPENMPI FUNCTIONALITY TEST (FORCED UCX/ROCE) ==="
echo "Testing OpenMPI features, performance, and collectives with forced UCX/RoCE..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca pml_base_verbose 100 \
  --mca pml_ucx_verbose 100 \
  --mca osc_ucx_verbose 100 \
  python mpi_test.py

echo "OpenMPI functionality test completed"

echo ""
echo "=== RUNNING OPENMPI NETWORK PERFORMANCE TEST (FORCED UCX/ROCE) ==="
echo "Testing OpenMPI network bandwidth, latency, and topology with forced UCX/RoCE..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca pml_base_verbose 100 \
  --mca pml_ucx_verbose 100 \
  python network_bandwidth_test_v2.py

echo "OpenMPI network performance test completed"

# Cleanup
rm -f hostfile.txt

echo "=== Test completed ==="
