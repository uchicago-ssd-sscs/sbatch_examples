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

# Activate conda environment (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate gpu || echo "Warning: could not activate conda env 'gpu'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

# Print SLURM environment
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Configure MPI to use 100 Gbps Ethernet bond1 interface
echo "=== CONFIGURING MPI FOR 100 GBPS ETHERNET (bond1) ==="

# Method 1: OpenMPI TCP/UCX configuration for Ethernet
export OMPI_MCA_btl="^openib"  # Disable InfiniBand BTLs
# Prefer UCX PML over legacy OB1
export OMPI_MCA_pml=ucx
# Force TCP to use the 10.150.0.0/16 subnet explicitly
export OMPI_MCA_btl_tcp_if_include="10.150.0.0/16"
export OMPI_MCA_oob_tcp_if_include="10.150.0.0/16"
# Optionally exclude slower subnets
export OMPI_MCA_btl_tcp_if_exclude="127.0.0.1/8,172.16.0.0/12,192.168.0.0/16"
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_rndv_eager_limit=65536
export OMPI_MCA_btl_tcp_rcvbuf=2097152
export OMPI_MCA_btl_tcp_sndbuf=2097152

# Method 2: UCX configuration for Ethernet
export UCX_NET_DEVICES="bond1"
export UCX_TLS="tcp"
export UCX_SOCKADDR_TLS_PRIORITY="tcp"
export UCX_TCP_CM_REUSEADDR=y
export UCX_TCP_MAX_RETRIES=10
export UCX_VERBOSE=1

# Method 3: Disable shared memory for cross-node communication
export OMPI_MCA_btl_sm="^sm"  # Disable shared memory BTL

# Print network interface information
echo "=== NETWORK INTERFACE INFORMATION ==="
echo "Available interfaces:"
ip -4 addr show bond1 || true

# Detect local bond1 IPv4 and bind UCX sockets to it
BOND1_IP=$(ip -4 -o addr show dev bond1 | awk '{print $4}' | cut -d/ -f1 | head -n1)
if [ -n "$BOND1_IP" ]; then
  export UCX_SOCKADDR_CM_SOURCE_ADDRESS="$BOND1_IP"
  echo "Local bond1 IPv4: $BOND1_IP"
else
  echo "Warning: could not detect bond1 IPv4"
fi

echo "Bond1 interface details (100 Gbps):"
ip addr show bond1 2>/dev/null || echo "bond1 not found"

echo "InfiniBand interfaces:"
ip link show | grep ib || echo "No InfiniBand interfaces found"

echo "=== MPI CONFIGURATION ==="
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
echo "OMPI_MCA_btl_tcp_if_include: ${OMPI_MCA_btl_tcp_if_include}"
echo "OMPI_MCA_oob_tcp_if_include: ${OMPI_MCA_oob_tcp_if_include}"
echo "UCX_NET_DEVICES: $UCX_NET_DEVICES"
echo "UCX_TLS: $UCX_TLS"
echo "UCX_SOCKADDR_CM_SOURCE_ADDRESS: ${UCX_SOCKADDR_CM_SOURCE_ADDRESS}"

# Test MPI
echo "Testing MPI multi-node communication with 100 Gbps Ethernet configuration..."

# Create hostfile using allocated hostnames (compatible with PRTE/SLURM)
echo "Creating hostfile (hostnames from SLURM allocation)..."
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

echo "Route to peer (debug):"
PEER_HOST=$(sed -n '2p' hostfile.txt)
PEER_IP=$(getent hosts "$PEER_HOST" | awk '{print $1}')
echo "Peer host: $PEER_HOST, resolves to: $PEER_IP"
ip route get "$PEER_IP" | cat

# Test MPI with 100 Gbps Ethernet configuration
echo "Testing MPI with 100 Gbps Ethernet configuration..."
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt \
  --mca pml ucx \
  --mca oob_tcp_if_include 10.150.0.0/16 \
  --mca btl_tcp_if_include 10.150.0.0/16 \
  --mca btl_tcp_if_exclude 127.0.0.1/8,172.16.0.0/12,192.168.0.0/16 \
  -x OMPI_MCA_btl -x OMPI_MCA_pml \
  -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include -x OMPI_MCA_btl_tcp_if_exclude \
  -x UCX_NET_DEVICES -x UCX_TLS -x UCX_SOCKADDR_TLS_PRIORITY -x UCX_SOCKADDR_CM_SOURCE_ADDRESS -x UCX_TCP_CM_REUSEADDR -x UCX_TCP_MAX_RETRIES -x UCX_VERBOSE \
  python mpi_test.py

echo "MPI test completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST WITH 100 GBPS ETHERNET ==="
echo "Testing actual cross-node network performance with 100 Gbps Ethernet..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt \
  --mca pml ucx \
  --mca oob_tcp_if_include 10.150.0.0/16 \
  --mca btl_tcp_if_include 10.150.0.0/16 \
  --mca btl_tcp_if_exclude 127.0.0.1/8,172.16.0.0/12,192.168.0.0/16 \
  -x OMPI_MCA_btl -x OMPI_MCA_pml \
  -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include -x OMPI_MCA_btl_tcp_if_exclude \
  -x UCX_NET_DEVICES -x UCX_TLS -x UCX_SOCKADDR_TLS_PRIORITY -x UCX_SOCKADDR_CM_SOURCE_ADDRESS -x UCX_TCP_CM_REUSEADDR -x UCX_TCP_MAX_RETRIES -x UCX_VERBOSE \
  python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

bash slurm_diagnostic.sh
