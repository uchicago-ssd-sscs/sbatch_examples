#!/usr/bin/env python3

import numpy as np
import time
import platform
import socket
import os
import sys

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    sys.exit(1)

def test_network_bandwidth(comm, rank, size, data_size_mb=4096, iterations=10):
    """Bidirectional network bandwidth test with comprehensive metrics"""
    
    if size < 2:
        return
    
    if rank == 0:
        print(f"Testing bidirectional {data_size_mb}MB bandwidth with {iterations} iterations")
    
    # Use larger data sizes to reduce overhead
    data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
    send_data = np.random.random(data_size).astype(np.float64)
    recv_data = np.empty(data_size, dtype=np.float64)
    
    # Warm up the network with bidirectional transfer
    if rank == 0:
        comm.Sendrecv(send_data, dest=1, sendtag=999,
                     recvbuf=recv_data, source=1, recvtag=1000)
    elif rank == 1:
        comm.Sendrecv(send_data, dest=0, sendtag=1000,
                     recvbuf=recv_data, source=0, recvtag=999)
    
    comm.Barrier()
    
    # Test bidirectional communication with comprehensive timing
    if rank == 0:
        print(f"Data size: {data_size} elements ({data_size * 8 / (1024*1024):.1f} MB)")
        
        # Bidirectional transfer
        start_time = MPI.Wtime()  # Use MPI timing for better accuracy
        for i in range(iterations):
            comm.Sendrecv(send_data, dest=1, sendtag=123,
                         recvbuf=recv_data, source=1, recvtag=124)
            if i % 5 == 0:  # Progress indicator
                print(f"  Bidirectional progress: {i+1}/{iterations}")
        end_time = MPI.Wtime()
        
        # Calculate bidirectional bandwidth
        bytes_transferred = data_size * 8 * iterations * 2  # Both directions
        bandwidth = bytes_transferred / ((end_time - start_time) * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Bidirectional {data_size_mb}MB: {bandwidth:.2f} GB/s")
        
    elif rank == 1:
        # Participate in bidirectional transfer
        for i in range(iterations):
            comm.Sendrecv(send_data, dest=0, sendtag=124,
                         recvbuf=recv_data, source=0, recvtag=123)

def test_bidirectional_bandwidth(comm, rank, size, data_size_mb=2048, iterations=5):
    """Test bidirectional bandwidth with non-blocking operations"""
    
    if size < 2:
        return
    
    if rank == 0:
        print(f"Testing bidirectional {data_size_mb}MB bandwidth with {iterations} iterations")
    
    data_size = int(data_size_mb * 1024 * 1024 // 8)
    send_data = np.random.random(data_size).astype(np.float64)
    recv_data = np.empty(data_size, dtype=np.float64)
    
    if rank in [0, 1]:
        other_rank = 1 if rank == 0 else 0
        
        # Test bidirectional with non-blocking operations
        start_time = MPI.Wtime()
        for _ in range(iterations):
            # Post receives first
            req_recv = comm.Irecv(recv_data, source=other_rank, tag=456)
            # Then send
            req_send = comm.Isend(send_data, dest=other_rank, tag=456)
            # Wait for both to complete
            req_recv.Wait()
            req_send.Wait()
        bidir_time = MPI.Wtime() - start_time
        
        # Calculate bidirectional bandwidth
        bytes_transferred = data_size * 8 * iterations * 2  # Both directions
        bidir_bandwidth = bytes_transferred / (bidir_time * 1024 * 1024 * 1024)  # GB/s
        bidir_bandwidth_mbps = bytes_transferred / (bidir_time * 1024 * 1024)  # MB/s
        
        print(f"Bidirectional {data_size_mb}MB: {bidir_bandwidth:.2f} GB/s")

def test_latency(comm, rank, size, iterations=10):
    """Test network latency with small messages"""
    
    if size < 2:
        return
    
    if rank == 0:
        print(f"Testing latency with {iterations} iterations")
    
    # Test with small messages to measure latency
    if rank == 0:
        start_time = MPI.Wtime()
        for i in range(iterations):
            comm.send(i, dest=1, tag=789)
        end_time = MPI.Wtime()
        
        total_time = end_time - start_time
        avg_latency = total_time / iterations * 1000000  # microseconds
        min_latency = total_time / iterations * 1000000  # approximation
        
        print(f"Average latency: {avg_latency:.2f} microseconds")
        print(f"Total time: {total_time:.6f} seconds")
        print(f"Messages sent: {iterations}")
        
    elif rank == 1:
        for i in range(iterations):
            data = comm.recv(source=0, tag=789)

def test_different_message_sizes(comm, rank, size):
    """Test bidirectional bandwidth with different message sizes"""
    
    if size < 2:
        return
    
    if rank == 0:
        print("Testing bidirectional bandwidth with different message sizes:")
    
    # Test different message sizes - expanded range for comprehensive testing
    message_sizes = [1024, 10240, 102400, 1048576, 10485760, 52428800, 104857600]  # 1KB, 10KB, 100KB, 1MB, 10MB, 50MB, 100MB
    
    for msg_size in message_sizes:
        data_size = msg_size // 8  # float64 = 8 bytes
        iterations = 1  # Single iteration for quick testing
        
        send_data = np.random.random(data_size).astype(np.float64)
        recv_data = np.empty(data_size, dtype=np.float64)
        
        comm.Barrier()  # Synchronize before timing
        
        if rank == 0:
            start_time = MPI.Wtime()
            for _ in range(iterations):
                comm.Sendrecv(send_data, dest=1, sendtag=999,
                             recvbuf=recv_data, source=1, recvtag=1000)
            end_time = MPI.Wtime()
            
            total_time = end_time - start_time
            bytes_transferred = data_size * 8 * iterations * 2  # Both directions
            bandwidth = bytes_transferred / (total_time * 1024 * 1024)  # MB/s (matching Cray MPICH)
            
            msg_size_mb = (msg_size * 8) / (1024 * 1024)  # Convert to MB
            print(f"Bidirectional bandwidth ({msg_size} elements, {msg_size_mb:.1f}MB): {bandwidth:.2f} MB/s")
            
        elif rank == 1:
            for _ in range(iterations):
                comm.Sendrecv(send_data, dest=0, sendtag=1000,
                             recvbuf=recv_data, source=0, recvtag=999)

def test_network_topology(comm, rank, size):
    """Test network topology and connectivity"""
    
    if rank == 0:
        print("Network topology:")
    
    # Gather hostnames to understand topology
    hostname = platform.node()
    all_hostnames = comm.gather(hostname, root=0)
    
    if rank == 0:
        unique_nodes = set(all_hostnames)
        print(f"Total processes: {size}")
        print(f"Unique nodes: {len(unique_nodes)}")
        print(f"Node distribution:")
        for node in sorted(unique_nodes):
            count = all_hostnames.count(node)
            print(f"  {node}: {count} processes")
        
        if len(unique_nodes) > 1:
            print("✅ Multi-node communication detected!")
        else:
            print("⚠️  Single-node communication only")
    
    # Test communication between different nodes
    if size >= 2:
        if rank == 0:
            # Send to all other ranks
            for dest in range(1, size):
                comm.send(f"Hello from {hostname}", dest=dest, tag=1000)
        else:
            # Receive from rank 0
            message = comm.recv(source=0, tag=1000)
            print(f"Rank {rank} on {hostname}: Received '{message}'")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = platform.node()
    

    
    # Run all network tests (expanded data sizes for comprehensive performance testing)
    test_network_bandwidth(comm, rank, size, data_size_mb=1, iterations=1)  # 1MB test
    comm.Barrier()
    
    test_network_bandwidth(comm, rank, size, data_size_mb=10, iterations=1)  # 10MB test
    comm.Barrier()
    
    test_network_bandwidth(comm, rank, size, data_size_mb=50, iterations=1)  # 50MB test
    comm.Barrier()
    
    test_network_bandwidth(comm, rank, size, data_size_mb=100, iterations=1)  # 100MB test
    comm.Barrier()
    
    test_network_bandwidth(comm, rank, size, data_size_mb=500, iterations=1)  # 500MB test
    comm.Barrier()
    
    test_network_bandwidth(comm, rank, size, data_size_mb=1000, iterations=1)  # 1GB test
    comm.Barrier()
    
    test_bidirectional_bandwidth(comm, rank, size, data_size_mb=500, iterations=1)  # 500MB test
    comm.Barrier()
    
    test_latency(comm, rank, size, iterations=10)  # Reduced from 100
    comm.Barrier()
    
    test_different_message_sizes(comm, rank, size)
    comm.Barrier()
    
    test_network_topology(comm, rank, size)
    comm.Barrier()
    

    
    MPI.Finalize()

if __name__ == "__main__":
    main()
