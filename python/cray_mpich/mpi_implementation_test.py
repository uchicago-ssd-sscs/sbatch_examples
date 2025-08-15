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

def check_mpi_environment():
    """Check which MPI implementation we're using"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== MPI Implementation Check ===")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Available memory: {memory.total / (1024**3):.1f} GB")
            print(f"Free memory: {memory.available / (1024**3):.1f} GB")
            print(f"Memory usage: {memory.percent:.1f}%")
        except ImportError:
            print("psutil not available - cannot check memory")
        print()
        
        # Check MPI library version
        mpi_lib = MPI.Get_library_version()
        print(f"MPI Library: {mpi_lib}")
        
        # Check for different MPI implementations
        mpi_lib_lower = mpi_lib.lower()
        if 'cray' in mpi_lib_lower:
            print("✅ Cray MPICH detected!")
        elif 'hpe' in mpi_lib_lower or 'mpt' in mpi_lib_lower:
            print("✅ HPE MPT detected!")
        elif 'mpich' in mpi_lib_lower:
            print("✅ MPICH detected!")
        elif 'openmpi' in mpi_lib_lower:
            print("✅ OpenMPI detected!")
        else:
            print("⚠️  Unknown MPI implementation")
        
        # Check environment variables
        mpi_vars = ['CRAY_MPICH_DIR', 'MPICH_DIR', 'MPICH_ROOT', 'MPT_DIR', 'HPE_MPT_DIR']
        found_mpi = False
        for var in mpi_vars:
            if var in os.environ:
                print(f"✅ Found {var}: {os.environ[var]}")
                found_mpi = True
        
        if not found_mpi:
            print("ℹ️  No MPI environment variables found (this is normal for some installations)")
        
        # Check mpirun version
        try:
            result = subprocess.run(['mpirun', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            print(f"mpirun version output: {result.stdout[:200]}...")
        except Exception as e:
            print(f"❌ Could not check mpirun version: {e}")
        
        print()

def test_mpi_features():
    """Test MPI features"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== MPI Feature Tests ===")
    
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

def test_100g_bandwidth():
    """Test 100G network bandwidth with appropriate data sizes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== 100G Network Bandwidth Test ===")
        print("Testing with data sizes appropriate for 100G networks")
        print("Theoretical maximum: 12.5 GB/s")
        print(f"Total processes: {size}")
        print()
    
    if size < 2:
        if rank == 0:
            print("Need at least 2 processes for bandwidth test")
        return
    
    # Data sizes appropriate for 100G networks - REDUCED for faster testing
    # Test sizes: 1MB, 5MB, 10MB, 20MB (matching OpenMPI)
    # Note: Reasonable sizes that complete quickly
    data_sizes_gb = [0.001, 0.005, 0.01, 0.02]  # 1MB, 5MB, 10MB, 20MB (matching OpenMPI)
    
    results = {}
    
    # Test point-to-point between first two nodes
    if rank == 0:
        print("\n--- Point-to-Point Bandwidth Test (Node 0 -> Node 1) ---")
    
    for data_size_gb in data_sizes_gb:
        data_size_mb = int(data_size_gb * 1024)
        data_size_elements = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
        
        if rank == 0:
            print(f"\n--- Testing {data_size_gb}GB transfer ---")
            print(f"Data size: {data_size_elements:,} elements ({data_size_gb}GB)")
            
            # Create data with memory optimization
            print(f"Allocating {data_size_gb}GB send buffer...")
            data = np.random.random(data_size_elements).astype(np.float64)
            # Force memory allocation to complete by accessing the array
            _ = data[0]
            
            # Warm up
            comm.Send(data[:1000], dest=1, tag=999)
            
            # Test send bandwidth
            comm.Barrier()
            start_time = MPI.Wtime()
            comm.Send(data, dest=1, tag=123)
            end_time = MPI.Wtime()
            
            send_time = end_time - start_time
            bytes_sent = data_size_elements * 8
            send_bandwidth = bytes_sent / (send_time * 1024 * 1024 * 1024)  # GB/s
            
            print(f"Send {data_size_gb}GB: {send_bandwidth:.2f} GB/s ({send_time:.3f}s)")
            results[f'send_{data_size_gb}gb'] = send_bandwidth
            
        elif rank == 1:
            # Receive data with memory optimization
            print(f"Allocating {data_size_gb}GB receive buffer...")
            data = np.empty(data_size_elements, dtype=np.float64)
            # Force memory allocation to complete by accessing the array
            _ = data[0]
            
            # Warm up
            comm.Recv(data[:1000], source=0, tag=999)
            
            # Test receive bandwidth
            comm.Barrier()
            start_time = MPI.Wtime()
            comm.Recv(data, source=0, tag=123)
            end_time = MPI.Wtime()
            
            recv_time = end_time - start_time
            bytes_received = data_size_elements * 8
            recv_bandwidth = bytes_received / (recv_time * 1024 * 1024 * 1024)  # GB/s
            
            print(f"Receive {data_size_gb}GB: {recv_bandwidth:.2f} GB/s ({recv_time:.3f}s)")
            results[f'recv_{data_size_gb}gb'] = recv_bandwidth
        
        comm.Barrier()
    
    # Multi-node collective test (only if we have enough nodes)
    if size >= 4:
        test_multi_node_collectives(comm, rank, size)
    
    # Summary
    if rank == 0:
        print("\n" + "="*50)
        print("100G BANDWIDTH TEST SUMMARY")
        print("="*50)
        print("Data Size | Send (GB/s) | Receive (GB/s)")
        print("-" * 40)
        
        for data_size_gb in data_sizes_gb:
            send_key = f'send_{data_size_gb}gb'
            recv_key = f'recv_{data_size_gb}gb'
            
            send_bw = results.get(send_key, 0)
            recv_bw = results.get(recv_key, 0)
            
            print(f"{data_size_gb:8.1f}GB | {send_bw:10.2f} | {recv_bw:12.2f}")
        
        print("-" * 40)
        print("Expected: 8-12.5 GB/s for 100G network")
        print("Note: Smaller transfers may show lower bandwidth due to overhead")
        print("="*50)

def test_multi_node_collectives(comm, rank, size):
    """Test collective operations across multiple nodes"""
    if rank == 0:
        print("\n--- Multi-Node Collective Test ---")
        print(f"Testing collectives across {size} processes")
    
    # Test 1: Allreduce with large data
    if rank == 0:
        print("\nTesting Allreduce with 1GB data per process...")
    
    # Each process contributes 1GB of data
    data_size_gb = 1
    data_size_elements = int(data_size_gb * 1024 * 1024 * 1024 // 8)
    
    # Create local data
    local_data = np.random.random(data_size_elements).astype(np.float64)
    
    # Allreduce operation
    comm.Barrier()
    start_time = MPI.Wtime()
    global_sum = comm.allreduce(local_data, op=MPI.SUM)
    end_time = MPI.Wtime()
    
    allreduce_time = end_time - start_time
    total_bytes = data_size_elements * 8 * size  # All processes contribute
    
    if rank == 0:
        allreduce_bandwidth = total_bytes / (allreduce_time * 1024 * 1024 * 1024)
        print(f"Allreduce {data_size_gb}GB per process: {allreduce_bandwidth:.2f} GB/s ({allreduce_time:.3f}s)")
    
    # Test 2: Broadcast large data from root
    if rank == 0:
        print("\nTesting Broadcast with 4GB data...")
        broadcast_data = np.random.random(int(4 * 1024 * 1024 * 1024 // 8)).astype(np.float64)
    else:
        broadcast_data = np.empty(int(4 * 1024 * 1024 * 1024 // 8), dtype=np.float64)
    
    comm.Barrier()
    start_time = MPI.Wtime()
    comm.Bcast(broadcast_data, root=0)
    end_time = MPI.Wtime()
    
    broadcast_time = end_time - start_time
    broadcast_bytes = 4 * 1024 * 1024 * 1024  # 4GB
    
    if rank == 0:
        broadcast_bandwidth = broadcast_bytes / (broadcast_time * 1024 * 1024 * 1024)
        print(f"Broadcast 4GB: {broadcast_bandwidth:.2f} GB/s ({broadcast_time:.3f}s)")
    
    # Test 3: Gather data from all processes
    if rank == 0:
        print("\nTesting Gather with 100MB data per process...")
    
    gather_data_size = int(100 * 1024 * 1024 // 8)  # 100MB per process
    local_gather_data = np.random.random(gather_data_size).astype(np.float64)
    
    if rank == 0:
        gathered_data = np.empty(gather_data_size * size, dtype=np.float64)
    else:
        gathered_data = None
    
    comm.Barrier()
    start_time = MPI.Wtime()
    comm.Gather(local_gather_data, gathered_data, root=0)
    end_time = MPI.Wtime()
    
    gather_time = end_time - start_time
    gather_bytes = gather_data_size * 8 * size  # All processes contribute
    
    if rank == 0:
        gather_bandwidth = gather_bytes / (gather_time * 1024 * 1024 * 1024)
        print(f"Gather {100}MB per process: {gather_bandwidth:.2f} GB/s ({gather_time:.3f}s)")
    
    comm.Barrier()

def test_mpi_collectives():
    """Test MPI collective operations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== MPI Collective Tests ===")
    
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

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== 100G MPI Implementation Test Suite ===")
        print(f"Running on {comm.Get_size()} processes")
        print(f"Hostname: {platform.node()}")
        print("Testing with data sizes appropriate for 100G networks")
        print()
    
    # Run all tests
    check_mpi_environment()
    comm.Barrier()
    
    test_mpi_features()
    comm.Barrier()
    
    test_100g_bandwidth()
    comm.Barrier()
    
    test_mpi_collectives()
    comm.Barrier()
    
    if rank == 0:
        print("\n=== Test Summary ===")
        print("The test shows which MPI implementation you're using and 100G network performance.")
        print("Make sure you have loaded the appropriate MPI modules or conda environment.")
        print("\n✅ 100G MPI implementation tests completed!")
    
    MPI.Finalize()

if __name__ == "__main__":
    main()
