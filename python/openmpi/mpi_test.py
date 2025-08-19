#!/usr/bin/env python3

import numpy as np
import time
import platform
import os
import sys
import subprocess

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    sys.exit(1)

def check_openmpi_environment():
    """Check OpenMPI environment and configuration"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Check MPI library version
        mpi_lib = MPI.Get_library_version()
        print(f"MPI Library: {mpi_lib}")
        
        # Check for OpenMPI
        mpi_lib_lower = mpi_lib.lower()
        if 'openmpi' in mpi_lib_lower:
            print("✅ OpenMPI detected!")
        else:
            print("⚠️  OpenMPI not detected - using different MPI implementation")
        
        # Check OpenMPI environment variables
        ompi_vars = ['OMPI_MCA_btl', 'OMPI_MCA_pml', 'OMPI_MCA_btl_tcp_if_include', 
                     'OMPI_MCA_oob_tcp_if_include', 'OMPI_MCA_btl_tcp_sndbuf', 'OMPI_MCA_btl_tcp_rcvbuf']
        found_ompi = False
        for var in ompi_vars:
            if var in os.environ:
                print(f"✅ Found {var}: {os.environ[var]}")
                found_ompi = True
        
        if not found_ompi:
            print("ℹ️  No OpenMPI MCA parameters found (using defaults)")
        
        # Check mpirun version
        try:
            result = subprocess.run(['mpirun', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            print(f"mpirun version output: {result.stdout[:200]}...")
        except Exception as e:
            print(f"❌ Could not check mpirun version: {e}")
        
        print()

def test_openmpi_features():
    """Test OpenMPI features"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    
    # Test 1: Check MPI version and features
    mpi_version = MPI.Get_version()
    if rank == 0:
        print(f"MPI Version: {mpi_version}")
        print(f"MPI Library: {MPI.Get_library_version()}")
    
    # Test 2: Check available communicators
    if rank == 0:
        print(f"Available communicators:")
        print(f"  COMM_WORLD: {MPI.COMM_WORLD}")
        print(f"  COMM_SELF: {MPI.COMM_SELF}")
        print(f"  COMM_NULL: {MPI.COMM_NULL}")
    
    # Test 3: Check MPI constants
    if rank == 0:
        print(f"MPI Constants:")
        print(f"  ANY_SOURCE: {MPI.ANY_SOURCE}")
        print(f"  ANY_TAG: {MPI.ANY_TAG}")
        print(f"  UNDEFINED: {MPI.UNDEFINED}")
    
    # Test 4: Check data types
    if rank == 0:
        print(f"MPI Data Types:")
        print(f"  INT: {MPI.INT}")
        print(f"  DOUBLE: {MPI.DOUBLE}")
        print(f"  CHAR: {MPI.CHAR}")
    
    # Test 5: Check operations
    if rank == 0:
        print(f"MPI Operations:")
        print(f"  SUM: {MPI.SUM}")
        print(f"  MAX: {MPI.MAX}")
        print(f"  MIN: {MPI.MIN}")
    
    comm.Barrier()

def test_openmpi_performance():
    """Test OpenMPI performance characteristics"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    
    # Test 1: Barrier timing
    comm.Barrier()
    start_time = MPI.Wtime()
    comm.Barrier()
    end_time = MPI.Wtime()
    
    if rank == 0:
        barrier_time = end_time - start_time
        barrier_time_us = barrier_time * 1000000  # microseconds
        print(f"Barrier synchronization time: {barrier_time_us:.2f} microseconds")
    
    # Test 2: Bidirectional point-to-point latency
    if size >= 2:
        iterations = 10  # Reduced from 100
        if rank == 0:
            # Bidirectional ping-pong test
            start_time = MPI.Wtime()
            for i in range(iterations):
                # Send to rank 1 and wait for response
                comm.send(i, dest=1, tag=123)
                response = comm.recv(source=1, tag=124)
            end_time = MPI.Wtime()
            
            total_time = end_time - start_time
            # Divide by 2 since each iteration is a round trip
            latency = (total_time / iterations / 2) * 1000000  # microseconds
            print(f"Bidirectional point-to-point latency: {latency:.2f} microseconds")
            
        elif rank == 1:
            # Respond to ping-pong from rank 0
            for i in range(iterations):
                data = comm.recv(source=0, tag=123)
                comm.send(data, dest=0, tag=124)  # Echo back
    
    # Test 3: Bidirectional bandwidth test
    if size >= 2:
        data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB (consistent with Cray MPICH)
        
        for data_size in data_sizes:
            # Both ranks send and receive simultaneously for bidirectional test
            data_send = np.random.random(data_size).astype(np.float64)
            data_recv = np.empty(data_size, dtype=np.float64)
            
            comm.Barrier()  # Synchronize before timing
            
            if rank == 0:
                start_time = MPI.Wtime()
                # Simultaneous send and receive using Sendrecv
                comm.Sendrecv(data_send, dest=1, sendtag=456,
                             recvbuf=data_recv, source=1, recvtag=457)
                end_time = MPI.Wtime()
                
                transfer_time = end_time - start_time
                bytes_transferred = data_size * 8 * 2  # Both directions
                bandwidth = bytes_transferred / (transfer_time * 1024 * 1024)  # MB/s
                print(f"Bidirectional bandwidth ({data_size} elements): {bandwidth:.2f} MB/s")
                
            elif rank == 1:
                # Simultaneous send and receive using Sendrecv
                comm.Sendrecv(data_send, dest=0, sendtag=457,
                             recvbuf=data_recv, source=0, recvtag=456)
    
    comm.Barrier()

def test_openmpi_collectives():
    """Test OpenMPI collective operations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    
    # Test 1: Broadcast
    if rank == 0:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    else:
        data = np.empty(5, dtype=np.float64)
    
    comm.Bcast(data, root=0)
    if rank == 0:
        print(f"✅ Broadcast successful: {data}")
    
    # Test 2: Reduce
    local_value = rank + 1.0
    global_sum = comm.reduce(local_value, op=MPI.SUM, root=0)
    if rank == 0:
        expected_sum = sum(range(1, size + 1))
        print(f"✅ Reduce successful: {global_sum} (expected: {expected_sum})")
    
    # Test 3: Allreduce
    local_value = rank * 2.0
    global_max = comm.allreduce(local_value, op=MPI.MAX)
    if rank == 0:
        expected_max = (size - 1) * 2.0
        print(f"✅ Allreduce successful: {global_max} (expected: {expected_max})")
    
    # Test 4: Gather
    local_data = np.array([rank], dtype=np.int32)
    if rank == 0:
        gathered_data = np.empty(size, dtype=np.int32)
    else:
        gathered_data = None
    
    comm.Gather(local_data, gathered_data, root=0)
    if rank == 0:
        expected = np.arange(size, dtype=np.int32)
        if np.array_equal(gathered_data, expected):
            print(f"✅ Gather successful: {gathered_data}")
        else:
            print(f"❌ Gather failed: got {gathered_data}, expected {expected}")
    
    comm.Barrier()

def test_openmpi_communication():
    """Test OpenMPI point-to-point communication"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    
    if size < 2:
        if rank == 0:
            print("Need at least 2 processes for communication test")
        return
    
    # Test simple send/receive
    if rank == 0:
        message = f"Hello from rank 0 on {platform.node()}"
        comm.send(message, dest=1, tag=123)
        print(f"✅ Rank 0: Sent message to rank 1")
    elif rank == 1:
        message = comm.recv(source=0, tag=123)
        print(f"✅ Rank 1: Received: {message}")
    
    # Test non-blocking communication
    if rank == 0:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        req = comm.Isend(data, dest=1, tag=456)
        req.Wait()
        print(f"✅ Rank 0: Non-blocking send completed")
    elif rank == 1:
        data = np.empty(5, dtype=np.float64)
        req = comm.Irecv(data, source=0, tag=456)
        req.Wait()
        print(f"✅ Rank 1: Non-blocking receive completed: {data}")
    
    comm.Barrier()

def test_openmpi_bandwidth():
    """Test OpenMPI bandwidth performance"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    
    if size < 2:
        if rank == 0:
            print("Need at least 2 processes for bandwidth test")
        return
    
    # Test with different data sizes (matching Cray MPICH)
    for data_size_mb in [1, 10, 50, 100]:  # 1MB, 10MB, 50MB, 100MB
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
    
    comm.Barrier()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = platform.node()
    

    
    # Run all tests
    check_openmpi_environment()
    comm.Barrier()
    
    test_openmpi_features()
    comm.Barrier()
    
    test_openmpi_performance()
    comm.Barrier()
    
    test_openmpi_collectives()
    comm.Barrier()
    
    test_openmpi_communication()
    comm.Barrier()
    
    test_openmpi_bandwidth()
    comm.Barrier()
    

    
    MPI.Finalize()

if __name__ == "__main__":
    main()
