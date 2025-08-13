#!/bin/bash
#SBATCH --job-name=Multi_GPU_Benchmark
#SBATCH --output=%j_multi_gpu_benchmark.out
#SBATCH --error=%j_multi_gpu_benchmark.err
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=L40S
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load necessary modules (adjust based on your system)
#module load python/anaconda3
#module load cuda/11.8
#module load openmpi/4.1.4

# Activate conda environment
conda init
conda activate gpu

# Configure MPI to use 100 Gbps Ethernet bond1 interface
echo "=== CONFIGURING MPI FOR 100 GBPS ETHERNET (bond1) ==="
export OMPI_MCA_btl="^openib"  # Disable InfiniBand BTLs
export OMPI_MCA_btl_tcp_if_include="bond1"  # Use bond1 for TCP
export OMPI_MCA_oob_tcp_if_include="bond1"  # Use bond1 for out-of-band
export OMPI_MCA_btl_tcp_eager_limit=65536
export OMPI_MCA_btl_tcp_rndv_eager_limit=65536
export OMPI_MCA_btl_tcp_rcvbuf=2097152
export OMPI_MCA_btl_tcp_sndbuf=2097152
export OMPI_MCA_btl_sm="^sm"  # Disable shared memory for cross-node
export UCX_NET_DEVICES="bond1"
export UCX_TLS="tcp"
export UCX_TCP_CM_REUSEADDR=y
export UCX_TCP_MAX_RETRIES=10

echo "MPI Configuration:"
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
echo "OMPI_MCA_btl_tcp_if_include: $OMPI_MCA_btl_tcp_if_include"
echo "OMPI_MCA_oob_tcp_if_include: $OMPI_MCA_oob_tcp_if_include"
echo "UCX_NET_DEVICES: $UCX_NET_DEVICES"
echo "UCX_TLS: $UCX_TLS"

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the GPU benchmark
echo "Starting Multi-GPU benchmark on multiple nodes"
echo "$(nvidia-smi)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the distributed MPI benchmark with proper node distribution
echo "Creating hostfile for MPI..."
# Expand SLURM node list and create proper hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
echo "Hostfile contents:"
cat hostfile.txt

echo "Launching MPI benchmark..."
echo "Launching $SLURM_NTASKS MPI processes across $SLURM_NNODES nodes..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python gpu_benchmark_mpi_multi_node.py

echo "Multi-GPU benchmark completed"

echo ""
echo "=== RUNNING NETWORK BANDWIDTH TEST ==="
echo "Testing actual cross-node network performance..."
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt python network_bandwidth_test_v2.py

echo "Network bandwidth test completed"

# Run diagnostic if needed
bash slurm_diagnostic.sh 