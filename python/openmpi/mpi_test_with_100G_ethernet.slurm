#!/bin/bash
#SBATCH --job-name=MPI_Test_100G_Config
#SBATCH --output=%j_mpi_test_100G.out
#SBATCH --error=%j_mpi_test_100G.err
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

# Print SLURM environment
echo "=== OpenMPI 100G Ethernet Test (Exclusive Node Allocation) ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Exclusive allocation: YES (no interference from other jobs)"
echo ""

# Configure MPI to use 100 Gbps Ethernet bond1 interface
echo "=== CONFIGURING MPI FOR 100 GBPS ETHERNET (bond1) ==="

# Allow OpenMPI to auto-detect and use the best available transport
# Removed all TCP-specific restrictions to enable optimal performance
echo "Allowing OpenMPI to auto-detect optimal transport (no TCP restrictions)"

# Optional: Set high-performance tuning parameters but don't force specific transport
export OMPI_MCA_btl_tcp_sndbuf=67108864    # Large send buffer if TCP is used
export OMPI_MCA_btl_tcp_rcvbuf=67108864    # Large receive buffer if TCP is used

# Clear any UCX restrictions to allow auto-detection
unset UCX_NET_DEVICES UCX_TLS UCX_SOCKADDR_TLS_PRIORITY UCX_TCP_CM_REUSEADDR UCX_TCP_MAX_RETRIES UCX_VERBOSE

# Remove network interface restrictions - let OpenMPI choose the best interface
# Removed: OMPI_MCA_btl_tcp_if_include, OMPI_MCA_oob_tcp_if_include, PRTE_MCA_oob_tcp_if_include

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

echo "=== MPI CONFIGURATION (AUTO-DETECT OPTIMAL TRANSPORT) ==="
echo "OMPI_MCA_btl: ${OMPI_MCA_btl:-auto-detect}"
echo "OMPI_MCA_pml: ${OMPI_MCA_pml:-auto-detect}"
echo "OMPI_MCA_btl_tcp_sndbuf: ${OMPI_MCA_btl_tcp_sndbuf}"
echo "OMPI_MCA_btl_tcp_rcvbuf: ${OMPI_MCA_btl_tcp_rcvbuf}"
echo "Network interface restrictions: REMOVED (auto-detect best interface)"

echo "=== Kernel TCP buffer limits (diagnostic) ==="
sysctl -n net.core.rmem_max 2>/dev/null | sed 's/^/net.core.rmem_max=/' || echo "net.core.rmem_max: unknown"
sysctl -n net.core.wmem_max 2>/dev/null | sed 's/^/net.core.wmem_max=/' || echo "net.core.wmem_max: unknown"
sysctl -n net.ipv4.tcp_rmem 2>/dev/null | sed 's/^/net.ipv4.tcp_rmem=/' || echo "net.ipv4.tcp_rmem: unknown"
sysctl -n net.ipv4.tcp_wmem 2>/dev/null | sed 's/^/net.ipv4.tcp_wmem=/' || echo "net.ipv4.tcp_wmem: unknown"
sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null | sed 's/^/net.ipv4.tcp_congestion_control=/' || echo "net.ipv4.tcp_congestion_control: unknown"
sysctl -n net.ipv4.tcp_window_scaling 2>/dev/null | sed 's/^/net.ipv4.tcp_window_scaling=/' || echo "net.ipv4.tcp_window_scaling: unknown"
echo "UCX_SOCKADDR_CM_SOURCE_ADDRESS: ${UCX_SOCKADDR_CM_SOURCE_ADDRESS}"

# Test MPI
echo "Testing MPI multi-node communication with 100 Gbps Ethernet configuration..."

# Create hostfile using allocated hostnames with .data1 suffix for MPI
echo "Creating hostfile (hostnames from SLURM allocation with .data1 suffix)..."
scontrol show hostnames $SLURM_JOB_NODELIST | sed 's/$/.data1/' > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

# Note: hostnames are for placement; data path is constrained by *if_include=bond1*
echo "=== Link details per node (bond1) ==="
srun -N $SLURM_NNODES -n $SLURM_NNODES bash -lc 'echo -n "$(hostname) "; ethtool bond1 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "no ethtool"; cat /proc/net/bonding/bond1 2>/dev/null | sed -n "1,30p" || true'

echo "Collecting bond1 IPv4s for direct tests..."
srun -N $SLURM_NNODES -n $SLURM_NNODES bash -lc "ip -4 -o addr show dev bond1 | tr -s ' ' | cut -d' ' -f4 | cut -d/ -f1" | sort -u > bond1_ips.txt
cat bond1_ips.txt || true

H0=$(sed -n '1p' hostfile.txt)
H1=$(sed -n '2p' hostfile.txt)
IP0=$(sed -n '1p' bond1_ips.txt)
IP1=$(sed -n '2p' bond1_ips.txt)
echo "Node0 $H0 bond1 IP: $IP0"
echo "Node1 $H1 bond1 IP: $IP1"

if [ -n "$IP0" ] && [ -n "$IP1" ]; then
  echo "Testing 10.150 reachability from $H0 to $IP1 (standard MTU)"
  srun -N1 -w $H0 bash -lc "ping -c 3 $IP1 | cat"
  echo "Testing 10.150 reachability from $H0 to $IP1 (jumbo MTU 8972)"
  srun -N1 -w $H0 bash -lc "ping -c 3 -M do -s 8972 $IP1 | cat || echo 'Jumbo ping failed'"
else
  echo "Could not determine bond1 IPs; skipping ping tests"
fi

# Test OpenMPI functionality with auto-detected optimal transport
echo "=== RUNNING OPENMPI FUNCTIONALITY TEST (AUTO-DETECT TRANSPORT) ==="
echo "Testing OpenMPI features, performance, and collectives with optimal transport..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca btl_base_verbose 100 \
  --mca oob_base_verbose 100 \
  python mpi_test.py

echo "OpenMPI functionality test completed"

echo ""
echo "=== RUNNING OPENMPI NETWORK PERFORMANCE TEST (AUTO-DETECT TRANSPORT) ==="
echo "Testing OpenMPI network bandwidth, latency, and topology with optimal transport..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca btl_base_verbose 100 \
  --mca oob_base_verbose 100 \
  python network_bandwidth_test_v2.py

echo "OpenMPI network performance test completed"

bash slurm_diagnostic.sh
