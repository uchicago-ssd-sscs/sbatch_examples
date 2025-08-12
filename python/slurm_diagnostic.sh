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
echo "   scontrol show hostnames \$SLURM_JOB_NODELIST:"
scontrol show hostnames $SLURM_JOB_NODELIST
echo ""

echo "4. SLURM Configuration:"
echo "   SLURM MPI configuration:"
scontrol show config | grep -i mpi || echo "   No MPI config found"
echo ""

echo "5. Available MPI:"
echo "   mpirun version:"
mpirun --version 2>/dev/null || echo "   mpirun not found"
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
