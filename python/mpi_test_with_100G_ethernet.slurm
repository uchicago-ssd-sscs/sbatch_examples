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

# Method 1: OpenMPI TCP configuration for Ethernet (no UCX)
export OMPI_MCA_btl=self,tcp        # use only loopback and TCP BTLs
export OMPI_MCA_pml=ob1             # default pml without UCX
# Force TCP to use the 100G bond1 interface explicitly
export OMPI_MCA_btl_tcp_if_include="bond1"
export OMPI_MCA_oob_tcp_if_include="bond1"
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_rndv_eager_limit=65536
export OMPI_MCA_btl_tcp_rcvbuf=2097152
export OMPI_MCA_btl_tcp_sndbuf=2097152

# Method 2: (UCX not available in this OpenMPI build) — do not configure UCX
unset UCX_NET_DEVICES UCX_TLS UCX_SOCKADDR_TLS_PRIORITY UCX_TCP_CM_REUSEADDR UCX_TCP_MAX_RETRIES UCX_VERBOSE

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
echo "OMPI_MCA_pml: $OMPI_MCA_pml"
echo "UCX_SOCKADDR_CM_SOURCE_ADDRESS: ${UCX_SOCKADDR_CM_SOURCE_ADDRESS}"

# Test MPI
echo "Testing MPI multi-node communication with 100 Gbps Ethernet configuration..."

# Create hostfile using allocated hostnames (compatible with PRTE/SLURM)
echo "Creating hostfile (hostnames from SLURM allocation)..."
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Note: hostnames are for placement; data path is constrained by *if_include=bond1*

# Test MPI with 100 Gbps Ethernet configuration
echo "Testing MPI with 100 Gbps Ethernet configuration..."
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt \
  --mca pml ob1 \
  --mca btl self,tcp \
  --mca oob_tcp_if_include bond1 \
  --mca btl_tcp_if_include bond1 \
  --mca btl_base_verbose 100 \
  --mca oob_base_verbose 100 \
  -x OMPI_MCA_btl -x OMPI_MCA_pml \
  -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include \
  python mpi_test.py

echo "MPI test completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST WITH 100 GBPS ETHERNET ==="
echo "Testing actual cross-node network performance with 100 Gbps Ethernet..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt \
  --mca pml ob1 \
  --mca btl self,tcp \
  --mca oob_tcp_if_include bond1 \
  --mca btl_tcp_if_include bond1 \
  --mca btl_base_verbose 100 \
  --mca oob_base_verbose 100 \
  -x OMPI_MCA_btl -x OMPI_MCA_pml \
  -x OMPI_MCA_btl_tcp_if_include -x OMPI_MCA_oob_tcp_if_include \
  python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

bash slurm_diagnostic.sh
