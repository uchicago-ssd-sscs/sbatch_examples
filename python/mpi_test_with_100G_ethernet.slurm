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

# Configure MPI to use 100 Gbps Ethernet bond1 interface
echo "=== CONFIGURING MPI FOR 100 GBPS ETHERNET (bond1) ==="

# Method 1: OpenMPI TCP BTL configuration for Ethernet
export OMPI_MCA_btl="^openib"  # Disable InfiniBand BTLs
export OMPI_MCA_btl_tcp_if_include="bond1"  # Use bond1 for TCP
export OMPI_MCA_oob_tcp_if_include="bond1"  # Use bond1 for out-of-band
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_rndv_eager_limit=65536
export OMPI_MCA_btl_tcp_rcvbuf=2097152
export OMPI_MCA_btl_tcp_sndbuf=2097152

# Method 2: UCX configuration for Ethernet
export UCX_NET_DEVICES="bond1"
export UCX_TLS="tcp"
export UCX_TCP_CM_REUSEADDR=y
export UCX_TCP_MAX_RETRIES=10
export UCX_VERBOSE=1

# Method 3: Disable shared memory for cross-node communication
export OMPI_MCA_btl_sm="^sm"  # Disable shared memory BTL

# Print network interface information
echo "=== NETWORK INTERFACE INFORMATION ==="
echo "Available interfaces:"
ip link show | grep -E "(bond1|ib)"

echo "Bond1 interface details (100 Gbps):"
ip addr show bond1 2>/dev/null || echo "bond1 not found"

echo "InfiniBand interfaces:"
ip link show | grep ib || echo "No InfiniBand interfaces found"

echo "=== MPI CONFIGURATION ==="
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
echo "OMPI_MCA_btl_tcp_if_include: $OMPI_MCA_btl_tcp_if_include"
echo "OMPI_MCA_oob_tcp_if_include: $OMPI_MCA_oob_tcp_if_include"
echo "UCX_NET_DEVICES: $UCX_NET_DEVICES"
echo "UCX_TLS: $UCX_TLS"

# Test MPI
echo "Testing MPI multi-node communication with 100 Gbps Ethernet configuration..."

# Create hostfile for mpirun
echo "Creating hostfile..."
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Test MPI with 100 Gbps Ethernet configuration
echo "Testing MPI with 100 Gbps Ethernet configuration..."
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python mpi_test.py

echo "MPI test completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST WITH 100 GBPS ETHERNET ==="
echo "Testing actual cross-node network performance with 100 Gbps Ethernet..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

bash slurm_diagnostic.sh
