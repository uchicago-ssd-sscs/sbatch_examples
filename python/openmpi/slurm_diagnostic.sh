#!/bin/bash

echo "=== SLURM DIAGNOSTIC INFORMATION ==="
echo ""

echo "1. SLURM Environment Variables:"
echo "   SLURM_JOB_ID: $SLURM_JOB_ID"
echo "   SLURM_NTASKS: $SLURM_NTASKS"
echo "   SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "   SLURM_NODELIST: $SLURM_NODELIST"
echo "   SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo ""

echo "2. Node Information:"
echo "   Current hostname: $(hostname)"
echo "   SLURM_NODEID: $SLURM_NODEID"
echo "   SLURM_PROCID: $SLURM_PROCID"
echo ""

echo "3. Expanded Node List:"
echo "   SLURM hostnames (scontrol show hostnames \$SLURM_JOB_NODELIST):"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""
echo "   MPI hostnames (from hostfile.txt with .data1 suffix):"
if [ -f "hostfile.txt" ]; then
    cat hostfile.txt
else
    echo "   hostfile.txt not found"
fi
echo ""

echo "4. SLURM Configuration:"
echo "   SLURM MPI configuration:"
scontrol show config | grep -i mpi || echo "   No MPI config found"
echo ""
echo "   SLURM MPI parameters:"
scontrol show config | grep -i "mpi.*param\|param.*mpi" || echo "   No MPI parameters found"
echo ""
echo "   SLURM plugins:"
scontrol show config | grep -i plugin || echo "   No plugin config found"
echo ""

echo "5. Available MPI:"
echo "   mpirun version:"
mpirun --version 2>/dev/null || echo "   mpirun not found"
echo ""
echo "   OpenMPI configuration:"
ompi_info --parsable 2>/dev/null | grep -E "^(mca|btl|pml)" | head -10 || echo "   ompi_info not available"
echo ""
echo "   Current OpenMPI MCA parameters:"
echo "   OMPI_MCA_btl: $OMPI_MCA_btl"
echo "   OMPI_MCA_pml: $OMPI_MCA_pml"
echo "   OMPI_MCA_btl_tcp_if_include: $OMPI_MCA_btl_tcp_if_include"
echo "   OMPI_MCA_btl_tcp_if_exclude: $OMPI_MCA_btl_tcp_if_exclude"
echo "   OMPI_MCA_btl_tcp_sndbuf: $OMPI_MCA_btl_tcp_sndbuf"
echo "   OMPI_MCA_btl_tcp_rcvbuf: $OMPI_MCA_btl_tcp_rcvbuf"
echo "   OMPI_MCA_btl_tcp_eager_limit: $OMPI_MCA_btl_tcp_eager_limit"
echo "   OMPI_MCA_btl_tcp_rndv_eager_limit: $OMPI_MCA_btl_tcp_rndv_eager_limit"
echo "   OMPI_MCA_btl_tcp_connect_timeout: $OMPI_MCA_btl_tcp_connect_timeout"
echo "   OMPI_MCA_btl_tcp_connect_retries: $OMPI_MCA_btl_tcp_connect_retries"
echo "   OMPI_MCA_btl_sm: $OMPI_MCA_btl_sm"
echo "   OMPI_MCA_oob_tcp_if_include: $OMPI_MCA_oob_tcp_if_include"
echo "   PRTE_MCA_oob_tcp_if_include: $PRTE_MCA_oob_tcp_if_include"
echo "   UCX_SOCKADDR_CM_SOURCE_ADDRESS: $UCX_SOCKADDR_CM_SOURCE_ADDRESS"
echo ""

echo "6. Network Connectivity:"
echo "   Testing connectivity to other nodes:"
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    if [ "$node" != "$(hostname)" ]; then
        echo "   Testing connection to $node..."
        ping -c 1 $node 2>/dev/null && echo "   ✅ $node reachable" || echo "   ❌ $node not reachable"
    fi
done
echo ""

echo "7. Process Information:"
echo "   PID: $$"
echo "   PPID: $PPID"
echo "   SLURM_TASK_PID: $SLURM_TASK_PID"
echo ""

echo "=== END DIAGNOSTIC ==="
