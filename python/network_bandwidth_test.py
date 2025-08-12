#!/usr/bin/env python3

import numpy as np
import time
import platform
import socket

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    exit(1)

def test_network_bandwidth(comm, rank, size, data_size_mb=1024, iterations=10):
    """Test actual network bandwidth between nodes"""
    
    # Only test if we have at least 2 processes
    if size < 2:
        return
    
    # Create data to send
    data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
    data = np.random.random(data_size).astype(np.float64)
    
    # Test point-to-point communication
    if rank == 0:
        print(f"Testing network bandwidth with {data_size_mb}MB data, {iterations} iterations")
        print(f"Data size: {data_size} elements ({data_size * 8 / (1024*1024):.1f} MB)")
        
        # Send data to rank 1
        start_time = time.time()
        for _ in range(iterations):
            comm.Send(data, dest=1, tag=123)
        send_time = time.time() - start_time
        
        # Calculate send bandwidth
        bytes_sent = data_size * 8 * iterations
        send_bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Send bandwidth: {send_bandwidth:.2f} GB/s")
        
    elif rank == 1:
        # Receive data from rank 0
        received_data = np.empty(data_size, dtype=np.float64)
        
        start_time = time.time()
        for _ in range(iterations):
            comm.Recv(received_data, source=0, tag=123)
        recv_time = time.time() - start_time
        
        # Calculate receive bandwidth
        bytes_received = data_size * 8 * iterations
        recv_bandwidth = bytes_received / (recv_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Receive bandwidth: {recv_bandwidth:.2f} GB/s")
    
    # Test bidirectional communication
    if size >= 4:
        if rank in [0, 1]:
            # Test bidirectional between ranks 0 and 1
            other_rank = 1 if rank == 0 else 0
            
            # Send and receive simultaneously
            start_time = time.time()
            for _ in range(iterations):
                if rank == 0:
                    comm.Send(data, dest=other_rank, tag=456)
                    comm.Recv(received_data, source=other_rank, tag=457)
                else:
                    comm.Recv(received_data, source=other_rank, tag=456)
                    comm.Send(data, dest=other_rank, tag=457)
            bidir_time = time.time() - start_time
            
            # Calculate bidirectional bandwidth
            bytes_transferred = data_size * 8 * iterations * 2  # Both directions
            bidir_bandwidth = bytes_transferred / (bidir_time * 1024 * 1024 * 1024)  # GB/s
            
            print(f"Bidirectional bandwidth: {bidir_bandwidth:.2f} GB/s")

def test_all_to_all_bandwidth(comm, rank, size, data_size_mb=256, iterations=5):
    """Test all-to-all communication bandwidth"""
    
    if size < 2:
        return
    
    # Create data to send
    data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
    send_data = np.random.random(data_size).astype(np.float64)
    recv_data = np.empty(data_size, dtype=np.float64)
    
    if rank == 0:
        print(f"\nTesting all-to-all bandwidth with {data_size_mb}MB data per process")
    
    # Test all-to-all communication
    start_time = time.time()
    for _ in range(iterations):
        comm.Alltoall(send_data, recv_data)
    alltoall_time = time.time() - start_time
    
    # Calculate total bandwidth
    total_bytes = data_size * 8 * size * iterations
    alltoall_bandwidth = total_bytes / (alltoall_time * 1024 * 1024 * 1024)  # GB/s
    
    if rank == 0:
        print(f"All-to-all bandwidth: {alltoall_bandwidth:.2f} GB/s total")
        print(f"Per-process bandwidth: {alltoall_bandwidth / size:.2f} GB/s")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hostname = platform.node()
    
    if rank == 0:
        print("=== NETWORK BANDWIDTH TEST ===")
        print(f"MPI Size: {size}")
        print(f"Testing between nodes: {hostname}")
    
    # Gather all hostnames
    all_hostnames = comm.gather(hostname, root=0)
    
    if rank == 0:
        unique_nodes = len(set(all_hostnames))
        print(f"Unique nodes: {unique_nodes}")
        print(f"Processes per node: {size // unique_nodes}")
        
        if unique_nodes > 1:
            print("✅ Testing cross-node network bandwidth")
        else:
            print("⚠️  All processes on same node - testing local communication")
    
    # Test different data sizes
    for data_size_mb in [64, 256, 1024]:
        test_network_bandwidth(comm, rank, size, data_size_mb, iterations=5)
        comm.Barrier()
    
    # Test all-to-all
    test_all_to_all_bandwidth(comm, rank, size, data_size_mb=64, iterations=3)
    
    if rank == 0:
        print("\n=== NETWORK BANDWIDTH TEST COMPLETED ===")
        print("Note: These are ACTUAL network bandwidth measurements")
        print("Compare with your InfiniBand specification (typically 100 Gbps = ~12.5 GB/s)")

if __name__ == "__main__":
    main()
