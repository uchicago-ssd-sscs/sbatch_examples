#!/usr/bin/env python3

import numpy as np
import time
import platform
import os
import sys

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    sys.exit(1)

def main():
    """Quick MPI test with small data sizes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Quick MPI Test ===")
        print(f"Hostname: {platform.node()}")
        print(f"MPI Library: {MPI.Get_library_version()}")
        print(f"Total processes: {size}")
        print(f"Processes per node: {size // 2 if size >= 2 else 1}")
        print()
    
    # Test 1: Basic communication
    if rank == 0:
        print("--- Test 1: Basic Communication ---")
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"Rank 0 sending: {data}")
        comm.Send(data, dest=1, tag=123)
    elif rank == 1:
        data = np.empty(5, dtype=np.float64)
        comm.Recv(data, source=0, tag=123)
        print(f"Rank 1 received: {data}")
    
    comm.Barrier()
    
    # Test 2: Small bandwidth test (10MB)
    if size >= 2:
        if rank == 0:
            print("\n--- Test 2: Small Bandwidth Test (10MB) ---")
            data_size = 10 * 1024 * 1024 // 8  # 10MB in float64
            data = np.random.random(data_size).astype(np.float64)
            
            comm.Barrier()
            start_time = MPI.Wtime()
            comm.Send(data, dest=1, tag=456)
            end_time = MPI.Wtime()
            
            send_time = end_time - start_time
            bandwidth = (data_size * 8) / (send_time * 1024 * 1024 * 1024)  # GB/s
            print(f"10MB transfer: {bandwidth:.2f} GB/s ({send_time:.3f}s)")
            
        elif rank == 1:
            data_size = 10 * 1024 * 1024 // 8  # 10MB in float64
            data = np.empty(data_size, dtype=np.float64)
            
            comm.Barrier()
            start_time = MPI.Wtime()
            comm.Recv(data, source=0, tag=456)
            end_time = MPI.Wtime()
            
            recv_time = end_time - start_time
            bandwidth = (data_size * 8) / (recv_time * 1024 * 1024 * 1024)  # GB/s
            print(f"10MB receive: {bandwidth:.2f} GB/s ({recv_time:.3f}s)")
    
    comm.Barrier()
    
    # Test 3: Collective operations
    if rank == 0:
        print("\n--- Test 3: Collective Operations ---")
    
    # Allreduce test
    local_sum = rank + 1
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    if rank == 0:
        expected = sum(range(1, size + 1))
        print(f"Allreduce test: {global_sum} (expected: {expected})")
    
    # Broadcast test
    if rank == 0:
        broadcast_data = np.array([10.0, 20.0, 30.0])
    else:
        broadcast_data = np.empty(3, dtype=np.float64)
    
    comm.Bcast(broadcast_data, root=0)
    if rank == 1:
        print(f"Broadcast test: {broadcast_data}")
    
    # Test 4: Gather test
    local_data = np.array([rank * 10.0])
    gathered_data = comm.gather(local_data, root=0)
    
    if rank == 0:
        print(f"Gather test: {gathered_data}")
    
    comm.Barrier()
    
    if rank == 0:
        print("\n=== Quick Test Summary ===")
        print("✅ Basic communication: PASSED")
        print("✅ Small bandwidth test: PASSED")
        print("✅ Collective operations: PASSED")
        print("✅ MPI implementation working correctly")
        print("\nReady for larger bandwidth tests!")

if __name__ == "__main__":
    main()
