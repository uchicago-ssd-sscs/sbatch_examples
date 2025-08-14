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

def test_cray_mpich_basic():
    """Basic Cray MPICH functionality test"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hostname = platform.node()
    
    print(f"Rank {rank}/{size} on {hostname}")
    print(f"Process ID: {os.getpid()}")
    
    # Test barrier synchronization
    comm.Barrier()
    if rank == 0:
        print("✅ Barrier synchronization successful")
    
    # Test gather operation
    all_hostnames = comm.gather(hostname, root=0)
    if rank == 0:
        unique_nodes = len(set(all_hostnames))
        print(f"✅ Gather operation successful")
        print(f"Total processes: {size}")
        print(f"Unique nodes: {unique_nodes}")
        print(f"Node distribution: {dict(zip(set(all_hostnames), [all_hostnames.count(x) for x in set(all_hostnames)]))}")

def test_cray_mpich_communication():
    """Test Cray MPICH point-to-point communication"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        print("Need at least 2 processes for communication test")
        return
    
    # Test simple send/receive
    if rank == 0:
        message = f"Hello from rank 0 on {platform.node()}"
        comm.send(message, dest=1, tag=123)
        print(f"Rank 0: Sent message to rank 1")
    elif rank == 1:
        message = comm.recv(source=0, tag=123)
        print(f"Rank 1: Received: {message}")
    
    # Test non-blocking communication
    if rank == 0:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        req = comm.Isend(data, dest=1, tag=456)
        req.Wait()
        print(f"Rank 0: Non-blocking send completed")
    elif rank == 1:
        data = np.empty(5, dtype=np.float64)
        req = comm.Irecv(data, source=0, tag=456)
        req.Wait()
        print(f"Rank 1: Non-blocking receive completed: {data}")

def test_cray_mpich_collectives():
    """Test Cray MPICH collective operations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Test broadcast
    if rank == 0:
        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    else:
        data = np.empty(3, dtype=np.float64)
    
    comm.Bcast(data, root=0)
    print(f"Rank {rank}: Broadcast received: {data}")
    
    # Test reduce
    local_value = rank + 1.0
    global_sum = comm.reduce(local_value, op=MPI.SUM, root=0)
    if rank == 0:
        expected_sum = sum(range(1, size + 1))
        print(f"Rank 0: Reduce sum = {global_sum} (expected: {expected_sum})")
    
    # Test allreduce
    local_value = rank * 2.0
    global_max = comm.allreduce(local_value, op=MPI.MAX)
    print(f"Rank {rank}: Allreduce max = {global_max}")

def test_cray_mpich_bandwidth():
    """Test Cray MPICH bandwidth performance"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        return
    
    # Test with different data sizes
    for data_size_mb in [1, 10, 100]:
        data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
        
        if rank == 0:
            data = np.random.random(data_size).astype(np.float64)
            start_time = MPI.Wtime()
            comm.Send(data, dest=1, tag=789)
            send_time = MPI.Wtime() - start_time
            
            bytes_sent = data_size * 8
            bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
            print(f"Send {data_size_mb}MB: {bandwidth:.2f} GB/s")
            
        elif rank == 1:
            data = np.empty(data_size, dtype=np.float64)
            start_time = MPI.Wtime()
            comm.Recv(data, source=0, tag=789)
            recv_time = MPI.Wtime() - start_time
            
            bytes_received = data_size * 8
            bandwidth = bytes_received / (recv_time * 1024 * 1024 * 1024)  # GB/s
            print(f"Receive {data_size_mb}MB: {bandwidth:.2f} GB/s")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== Cray MPICH Python Test Suite ===")
        print(f"MPI Implementation: {MPI.Get_library_version()}")
        print(f"MPI Version: {MPI.Get_version()}")
        print()
    
    # Run all tests
    test_cray_mpich_basic()
    comm.Barrier()
    
    test_cray_mpich_communication()
    comm.Barrier()
    
    test_cray_mpich_collectives()
    comm.Barrier()
    
    test_cray_mpich_bandwidth()
    comm.Barrier()
    
    if rank == 0:
        print("\n✅ All Cray MPICH tests completed successfully!")
    
    MPI.Finalize()

if __name__ == "__main__":
    main()
