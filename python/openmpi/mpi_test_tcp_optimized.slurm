#!/bin/bash
#SBATCH --job-name=MPI_Test_TCP_Optimized
#SBATCH --output=%j_mpi_test_tcp_optimized.out
#SBATCH --error=%j_mpi_test_tcp_optimized.err
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --partition=CPU
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate openmpi || echo "Warning: could not activate conda env 'openmpi'"
fi

echo "=== OPTIMIZED TCP CONFIGURATION (NO RDMA) ==="

# Aggressive TCP optimization for 100G Ethernet
export OMPI_MCA_btl=self,tcp             # Use only TCP
export OMPI_MCA_pml=ob1                  # Use ob1 PML
export OMPI_MCA_btl_tcp_sndbuf=134217728 # 128MB send buffer
export OMPI_MCA_btl_tcp_rcvbuf=134217728 # 128MB receive buffer
export OMPI_MCA_btl_tcp_eager_limit=131072    # 128KB eager
export OMPI_MCA_btl_tcp_rndv_eager_limit=131072 # 128KB rendezvous
export OMPI_MCA_btl_tcp_max_send_size=134217728  # 128MB max send

# Force use of 100G interface
export OMPI_MCA_btl_tcp_if_include="10.150.0.0/16"
export OMPI_MCA_oob_tcp_if_include="10.150.0.0/16"

# Disable competing transports
export OMPI_MCA_pml_ucx_priority=0       # Disable UCX
export OMPI_MCA_btl_vader_single_copy_mechanism=none

echo "=== TCP CONFIGURATION ==="
echo "Send/Recv buffers: 128MB each"
echo "Eager limits: 128KB"
echo "Max send: 128MB"
echo "Interface: 10.150.x.x only"

# Create hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.txt
cat hostfile.txt

echo ""
echo "=== TESTING OPTIMIZED TCP PERFORMANCE ==="
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  --mca btl_base_verbose 50 \
  python mpi_test.py

echo ""
echo "=== NETWORK BANDWIDTH TEST ==="
mpirun -np $SLURM_NTASKS --hostfile hostfile.txt --bind-to none \
  python network_bandwidth_test_v2.py

rm -f hostfile.txt
echo "=== Test completed ==="
