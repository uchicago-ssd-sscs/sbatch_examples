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

def check_cray_mpich_environment():
    """Check Cray MPICH environment and configuration"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== Cray MPICH Environment Check ===")
        print(f"MPI Library: {MPI.Get_library_version()}")
        print(f"MPI Version: {MPI.Get_version()}")
        
        # Check for Cray-specific environment variables
        cray_vars = ['CRAY_MPICH_DIR', 'MPICH_VERSION', 'MPICH_CC', 'MPICH_CXX']
        for var in cray_vars:
            value = os.environ.get(var, 'Not set')
            print(f"{var}: {value}")
        
        # Check mpirun version
        try:
            import subprocess
            result = subprocess.run(['mpirun', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"mpirun version: {result.stdout.strip().split()[2]}")
        except:
            print("mpirun version: Could not determine")

def test_cray_mpich_features():
    """Test Cray MPICH features and capabilities"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Cray MPICH Features Test ===")
        print(f"MPI Version: {MPI.Get_version()}")
        print(f"Communicator size: {size}")
        print(f"Available constants: MPI_SUM={MPI.SUM}, MPI_MAX={MPI.MAX}")
        print(f"Available data types: MPI_DOUBLE={MPI.DOUBLE}, MPI_INT={MPI.INT}")
        print(f"Available operations: MPI_SUM={MPI.SUM}, MPI_MAX={MPI.MAX}, MPI_MIN={MPI.MIN}")

def test_cray_mpich_performance():
    """Test Cray MPICH performance characteristics"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Cray MPICH Performance Tests ===")
    
    # Test 1: Barrier timing
    iterations = 20  # Reduced from 100
    comm.Barrier()
    start_time = MPI.Wtime()
    for _ in range(iterations):
        comm.Barrier()
    end_time = MPI.Wtime()
    
    if rank == 0:
        avg_barrier_time = (end_time - start_time) / iterations * 1000000  # microseconds
        print(f"Average barrier time ({iterations} iterations): {avg_barrier_time:.2f} microseconds")
    
    # Test 2: Point-to-point latency
    if size >= 2:
        iterations = 100  # Reduced from 1000
        if rank == 0:
            data = np.array([1.0], dtype=np.float64)
            start_time = MPI.Wtime()
            for _ in range(iterations):
                comm.Send(data, dest=1, tag=999)
            end_time = MPI.Wtime()
            avg_latency = (end_time - start_time) / iterations * 1000000  # microseconds
            print(f"Point-to-point latency ({iterations} iterations): {avg_latency:.2f} microseconds")
        elif rank == 1:
            data = np.empty(1, dtype=np.float64)
            for _ in range(iterations):
                comm.Recv(data, source=0, tag=999)
    
    # Test 3: Bandwidth test
    if size >= 2:
        data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB (consistent with OpenMPI)
        for data_size in data_sizes:
            if rank == 0:
                data = np.random.random(data_size).astype(np.float64)
                start_time = MPI.Wtime()
                comm.Send(data, dest=1, tag=888)
                end_time = MPI.Wtime()
                
                bytes_sent = data_size * 8
                bandwidth = bytes_sent / (end_time - start_time) / (1024 * 1024)  # MB/s
                print(f"Bandwidth ({data_size} elements): {bandwidth:.2f} MB/s")
            elif rank == 1:
                data = np.empty(data_size, dtype=np.float64)
                comm.Recv(data, source=0, tag=888)

def test_cray_mpich_collectives():
    """Test Cray MPICH collective operations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Cray MPICH Collective Operations Test ===")
    
    # Test broadcast
    if rank == 0:
        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    else:
        data = np.empty(3, dtype=np.float64)
    
    comm.Bcast(data, root=0)
    if rank == 0:
        print(f"Broadcast test: {data}")
    
    # Test reduce
    local_value = rank + 1.0
    global_sum = comm.reduce(local_value, op=MPI.SUM, root=0)
    if rank == 0:
        expected_sum = sum(range(1, size + 1))
        print(f"Reduce sum test: {global_sum} (expected: {expected_sum})")
    
    # Test allreduce
    local_value = rank * 2.0
    global_max = comm.allreduce(local_value, op=MPI.MAX)
    if rank == 0:
        print(f"Allreduce max test: {global_max}")
    
    # Test gather
    local_data = np.array([rank], dtype=np.float64)
    gathered_data = comm.gather(local_data, root=0)
    if rank == 0:
        print(f"Gather test: {gathered_data}")

def test_cray_mpich_communication():
    """Test Cray MPICH communication patterns"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Cray MPICH Communication Test ===")
    
    # Test simple send/receive
    if size >= 2:
        if rank == 0:
            message = f"Hello from rank 0 on {platform.node()}"
            comm.send(message, dest=1, tag=123)
            print(f"Simple send/receive: Message sent to rank 1")
        elif rank == 1:
            message = comm.recv(source=0, tag=123)
            print(f"Simple send/receive: Received: {message}")
    
    # Test non-blocking communication
    if size >= 2:
        if rank == 0:
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
            req = comm.Isend(data, dest=1, tag=456)
            req.Wait()
            print(f"Non-blocking communication: Send completed")
        elif rank == 1:
            data = np.empty(5, dtype=np.float64)
            req = comm.Irecv(data, source=0, tag=456)
            req.Wait()
            print(f"Non-blocking communication: Receive completed: {data}")

def test_cray_mpich_bandwidth():
    """Test Cray MPICH bandwidth performance"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        return
    
    if rank == 0:
        print("=== Cray MPICH Bandwidth Test ===")
    
    # Test with different data sizes
    for data_size_mb in [1, 5, 10, 20]:  # 1MB, 5MB, 10MB, 20MB
        data_size = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
        
        if rank == 0:
            print(f"Testing {data_size_mb}MB bandwidth with 1 iterations")
            print(f"Data size: {data_size} elements ({data_size * 8 / (1024*1024):.1f} MB)")
            
            data = np.random.random(data_size).astype(np.float64)
            start_time = MPI.Wtime()
            comm.Send(data, dest=1, tag=789)
            send_time = MPI.Wtime() - start_time
            
            bytes_sent = data_size * 8
            bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
            print(f"  Send progress: 1/1")
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
    size = comm.Get_size()
    hostname = platform.node()
    
    if rank == 0:
        print("=== Cray MPICH Python Test Suite ===")
        print(f"MPI Implementation: {MPI.Get_library_version()}")
        print(f"Running on {size} processes")
        print(f"Hostname: {hostname}")
        print()
    
    # Run all tests
    check_cray_mpich_environment()
    comm.Barrier()
    
    test_cray_mpich_features()
    comm.Barrier()
    
    test_cray_mpich_performance()
    comm.Barrier()
    
    test_cray_mpich_collectives()
    comm.Barrier()
    
    test_cray_mpich_communication()
    comm.Barrier()
    
    test_cray_mpich_bandwidth()
    comm.Barrier()
    
    if rank == 0:
        print("\n=== Test Summary ===")
        print("The test shows Cray MPICH implementation details and performance metrics.")
        print("Make sure you have loaded the appropriate Cray MPICH modules.")
        print("\n✅ Cray MPICH implementation tests completed!")
    
    MPI.Finalize()

if __name__ == "__main__":
    main()
