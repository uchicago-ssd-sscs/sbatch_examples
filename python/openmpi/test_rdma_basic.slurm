#!/bin/bash
#SBATCH --job-name=Test_RDMA_Basic
#SBATCH --output=%j_test_rdma_basic.out
#SBATCH --error=%j_test_rdma_basic.err
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --partition=CPU

echo "=== RDMA Infrastructure Test ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

echo ""
echo "=== Hardware Detection ==="
echo "PCI devices (Mellanox):"
lspci | grep -i mellanox || echo "No Mellanox devices found"

echo ""
echo "=== RDMA Tools Test ==="
echo "Testing ibv_devinfo:"
if command -v ibv_devinfo &> /dev/null; then
    ibv_devinfo || echo "ibv_devinfo failed"
else
    echo "❌ ibv_devinfo not installed"
fi

echo ""
echo "Testing ucx_info:"
if command -v ucx_info &> /dev/null; then
    ucx_info -d || echo "ucx_info failed"
else
    echo "❌ ucx_info not installed"
fi

echo ""
echo "=== Network Interface Analysis ==="
echo "Network interfaces:"
ip link show | grep -E "(bond|ib|mlx|eth)"

echo ""
echo "Bond1 details (if exists):"
if ip link show bond1 &> /dev/null; then
    ip addr show bond1
    ethtool bond1 | grep -E "(Speed|Duplex|Link)"
else
    echo "Bond1 interface not found"
fi

echo ""
echo "=== OpenMPI UCX Support Test ==="
if command -v ompi_info &> /dev/null; then
    echo "OpenMPI UCX components:"
    ompi_info --parsable --all | grep -i ucx || echo "No UCX support found"
else
    echo "❌ ompi_info not available"
fi

echo ""
echo "=== Test completed ==="
