#!/bin/bash
#SBATCH --job-name=MPI_Test_IB_Config
#SBATCH --output=%j_mpi_test_ib.out
#SBATCH --error=%j_mpi_test_ib.err
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --partition=L40S
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

# Configure MPI to use InfiniBand bond0 interface
echo "=== CONFIGURING MPI FOR INFINIBAND ==="

# Method 1: OpenMPI BTL configuration
export OMPI_MCA_btl_openib_if_include="bond0"
export OMPI_MCA_btl="^openib"  # Disable other BTLs
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_verbose=1

# Method 2: UCX configuration (alternative)
export UCX_NET_DEVICES="bond0"
export UCX_TLS="rc,ud"
export UCX_VERBOSE=1

# Method 3: Force specific network interface
export OMPI_MCA_oob_tcp_if_include="bond0"
export OMPI_MCA_btl_tcp_if_include="bond0"

# Method 4: Disable shared memory for cross-node communication
export OMPI_MCA_btl_sm="^sm"  # Disable shared memory BTL

# Print network interface information
echo "=== NETWORK INTERFACE INFORMATION ==="
echo "Available interfaces:"
ip link show | grep -E "(bond0|ib|eth|eno)"

echo "Bond0 interface details:"
ip addr show bond0 2>/dev/null || echo "bond0 not found"

echo "InfiniBand interfaces:"
ip link show | grep ib || echo "No InfiniBand interfaces found"

echo "=== MPI CONFIGURATION ==="
echo "OMPI_MCA_btl_openib_if_include: $OMPI_MCA_btl_openib_if_include"
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
echo "UCX_NET_DEVICES: $UCX_NET_DEVICES"
echo "OMPI_MCA_oob_tcp_if_include: $OMPI_MCA_oob_tcp_if_include"

# Test MPI
echo "Testing MPI multi-node communication with InfiniBand configuration..."

# Create hostfile for mpirun
echo "Creating hostfile..."
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test MPI with InfiniBand configuration
echo "Testing MPI with InfiniBand configuration..."
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python mpi_test.py

echo "MPI test completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST WITH IB CONFIG ==="
echo "Testing actual cross-node network performance with InfiniBand..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

bash slurm_diagnostic.sh
