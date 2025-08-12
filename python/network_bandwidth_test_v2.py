#!/usr/bin/env python3

import numpy as np
import time
import platform
import socket
import os

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    exit(1)

def test_network_bandwidth_improved(comm, rank, size, data_size_mb=4096, iterations=10):
    """Improved network bandwidth test with better methodology"""
    
    if size < 2:
        return
    
    # Use larger data sizes to reduce overhead
    data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
    data = np.random.random(data_size).astype(np.float64)
    
    # Warm up the network
    if rank == 0:
        comm.Send(data, dest=1, tag=999)
    elif rank == 1:
        received_data = np.empty(data_size, dtype=np.float64)
        comm.Recv(received_data, source=0, tag=999)
    
    comm.Barrier()
    
    # Test point-to-point communication with better timing
    if rank == 0:
        print(f"Testing network bandwidth with {data_size_mb}MB data, {iterations} iterations")
        print(f"Data size: {data_size} elements ({data_size * 8 / (1024*1024):.1f} MB)")
        
        # Send data to rank 1
        start_time = MPI.Wtime()  # Use MPI timing for better accuracy
        for i in range(iterations):
            comm.Send(data, dest=1, tag=123)
            if i % 5 == 0:  # Progress indicator
                print(f"  Send progress: {i+1}/{iterations}")
        comm.Barrier()  # Wait for all sends to complete
        send_time = MPI.Wtime() - start_time
        
        # Calculate send bandwidth
        bytes_sent = data_size * 8 * iterations
        send_bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Send bandwidth: {send_bandwidth:.2f} GB/s")
        print(f"Send time: {send_time:.3f} seconds")
        
    elif rank == 1:
        # Receive data from rank 0
        received_data = np.empty(data_size, dtype=np.float64)
        
        start_time = MPI.Wtime()
        for i in range(iterations):
            comm.Recv(received_data, source=0, tag=123)
        recv_time = MPI.Wtime() - start_time
        
        # Calculate receive bandwidth
        bytes_received = data_size * 8 * iterations
        recv_bandwidth = bytes_received / (recv_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Receive bandwidth: {recv_bandwidth:.2f} GB/s")
        print(f"Receive time: {recv_time:.3f} seconds")

def test_bidirectional_bandwidth(comm, rank, size, data_size_mb=2048, iterations=5):
    """Test bidirectional bandwidth with non-blocking operations"""
    
    if size < 2:
        return
    
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
        
        print(f"Bidirectional bandwidth: {bidir_bandwidth:.2f} GB/s")
        print(f"Bidirectional time: {bidir_time:.3f} seconds")

def test_large_transfer(comm, rank, size, data_size_mb=16384, iterations=3):
    """Test with very large data transfer to minimize overhead"""
    
    if size < 2:
        return
    
    data_size = int(data_size_mb * 1024 * 1024 // 8)
    
    if rank == 0:
        print(f"\nTesting large transfer: {data_size_mb}MB data, {iterations} iterations")
        data = np.random.random(data_size).astype(np.float64)
        
        start_time = MPI.Wtime()
        for i in range(iterations):
            comm.Send(data, dest=1, tag=789)
            print(f"  Large transfer progress: {i+1}/{iterations}")
        send_time = MPI.Wtime() - start_time
        
        bytes_sent = data_size * 8 * iterations
        large_bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Large transfer bandwidth: {large_bandwidth:.2f} GB/s")
        print(f"Large transfer time: {send_time:.3f} seconds")
        
    elif rank == 1:
        received_data = np.empty(data_size, dtype=np.float64)
        
        start_time = MPI.Wtime()
        for _ in range(iterations):
            comm.Recv(received_data, source=0, tag=789)
        recv_time = MPI.Wtime() - start_time
        
        bytes_received = data_size * 8 * iterations
        large_bandwidth = bytes_received / (recv_time * 1024 * 1024 * 1024)  # GB/s
        
        print(f"Large transfer bandwidth: {large_bandwidth:.2f} GB/s")
        print(f"Large transfer time: {recv_time:.3f} seconds")

def check_network_info(comm, rank):
    """Check network interface information"""
    
    if rank == 0:
        print("\n=== NETWORK INTERFACE INFORMATION ===")
        
        # Check for InfiniBand interfaces
        try:
            import subprocess
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
            if 'ib' in result.stdout.lower():
                print("✅ InfiniBand interfaces detected")
            else:
                print("⚠️  No InfiniBand interfaces found")
                
            # Check for high-speed interfaces
            if '100g' in result.stdout.lower() or '100000' in result.stdout.lower():
                print("✅ 100G interfaces detected")
            elif '40g' in result.stdout.lower() or '40000' in result.stdout.lower():
                print("✅ 40G interfaces detected")
            else:
                print("⚠️  No high-speed interfaces detected")
                
        except Exception as e:
            print(f"Could not check network interfaces: {e}")
        
        # Check MPI implementation
        print(f"MPI Implementation: {MPI.Get_library_version()}")
        
        # Check for environment variables
        ib_vars = ['OMPI_MCA_btl_openib_if_include', 'OMPI_MCA_btl', 'UCX_NET_DEVICES']
        for var in ib_vars:
            if var in os.environ:
                print(f"✅ {var}: {os.environ[var]}")
            else:
                print(f"⚠️  {var}: Not set")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hostname = platform.node()
    
    if rank == 0:
        print("=== IMPROVED NETWORK BANDWIDTH TEST ===")
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
    
    # Check network information
    check_network_info(comm, rank)
    
    # Test with different data sizes (larger to reduce overhead)
    for data_size_mb in [1024, 4096, 8192]:
        test_network_bandwidth_improved(comm, rank, size, data_size_mb, iterations=5)
        comm.Barrier()
    
    # Test bidirectional
    test_bidirectional_bandwidth(comm, rank, size, data_size_mb=2048, iterations=3)
    
    # Test very large transfer
    test_large_transfer(comm, rank, size, data_size_mb=16384, iterations=2)
    
    if rank == 0:
        print("\n=== IMPROVED NETWORK BANDWIDTH TEST COMPLETED ===")
        print("Expected bandwidth for 100 Gbps InfiniBand: ~12.5 GB/s")
        print("If results are significantly lower, check:")
        print("1. InfiniBand driver installation")
        print("2. MPI configuration (OpenMPI/UCX)")
        print("3. Network interface selection")
        print("4. System resource limits")

if __name__ == "__main__":
    main()
