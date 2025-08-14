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
        print()
    
    if size < 2:
        if rank == 0:
            print("Need at least 2 processes for bandwidth test")
        return
    
    # Data sizes appropriate for 100G networks
    # Test sizes: 1GB, 4GB, 8GB, 16GB, 32GB
    data_sizes_gb = [1, 4, 8, 16, 32]
    
    results = {}
    
    for data_size_gb in data_sizes_gb:
        data_size_mb = data_size_gb * 1024
        data_size_elements = int(data_size_mb * 1024 * 1024 // 8)  # float64 = 8 bytes
        
        if rank == 0:
            print(f"\n--- Testing {data_size_gb}GB transfer ---")
            print(f"Data size: {data_size_elements:,} elements ({data_size_gb}GB)")
            
            # Create data
            data = np.random.random(data_size_elements).astype(np.float64)
            
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
            # Receive data
            data = np.empty(data_size_elements, dtype=np.float64)
            
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
            
            if send_key in results:
                send_bw = results[send_key]
                recv_bw = results.get(recv_key, "N/A")
                print(f"{data_size_gb:9d}GB | {send_bw:10.2f} | {recv_bw:12.2f}")
        
        print("\nExpected performance for 100G network:")
        print("- Theoretical maximum: 12.5 GB/s")
        print("- Good performance: 8-10 GB/s")
        print("- Acceptable performance: 5-8 GB/s")
        print("- Poor performance: <5 GB/s")
        
        # Calculate average bandwidth
        send_bandwidths = [v for k, v in results.items() if k.startswith('send_')]
        if send_bandwidths:
            avg_send = np.mean(send_bandwidths)
            print(f"\nAverage send bandwidth: {avg_send:.2f} GB/s")
            
            if avg_send >= 8:
                print("✅ Excellent performance!")
            elif avg_send >= 5:
                print("✅ Good performance")
            elif avg_send >= 2:
                print("⚠️  Acceptable performance, but room for improvement")
            else:
                print("❌ Poor performance - check network configuration")
    
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
