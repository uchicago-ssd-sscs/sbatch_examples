#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
import platform
import psutil

# Try to import MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    print("MPI available - running distributed benchmark")
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available - cannot run distributed benchmark")
    sys.exit(1)

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_GPU_AVAILABLE = True
        print("PyTorch GPU available")
    else:
        TORCH_GPU_AVAILABLE = False
        print("PyTorch available but no GPU detected")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_GPU_AVAILABLE = False
    print("PyTorch not available")

def get_system_info():
    """Get detailed system information"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'mpi_rank': rank,
        'mpi_size': size,
    }
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = []
        info['gpu_memories'] = []
        
        for i in range(info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            info['gpu_names'].append(gpu_name)
            info['gpu_memories'].append(gpu_memory)
    
    return info

def benchmark_distributed_matrix_operations(size=8192, iterations=3):
    """Benchmark matrix operations using MPI across multiple nodes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_mpi = comm.Get_size()
    
    results = {}
    
    # CPU matrix operations (each node does its own)
    if rank == 0:
        print(f"Running distributed matrix operations with {size_mpi} nodes, size {size}x{size}")
    
    print(f"Rank {rank}: Running CPU matrix operations with size {size}x{size}")
    a_cpu = np.random.random((size, size)).astype(np.float32)
    b_cpu = np.random.random((size, size)).astype(np.float32)
    
    # Matrix multiplication
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    results['cpu_matmul_time'] = cpu_time / iterations
    results['cpu_matmul_gflops'] = (2 * size**3) / (cpu_time / iterations) / 1e9
    
    # Gather CPU results from all nodes
    all_cpu_times = comm.gather(results['cpu_matmul_time'], root=0)
    all_cpu_gflops = comm.gather(results['cpu_matmul_gflops'], root=0)
    
    # GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        gpu_count = torch.cuda.device_count()
        print(f"Rank {rank}: Running GPU matrix operations with {gpu_count} GPUs")
        
        # Test single GPU performance on this node
        torch.cuda.set_device(0)
        a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        single_gpu_time = time.time() - start_time
        results['single_gpu_time'] = single_gpu_time / iterations
        results['single_gpu_gflops'] = (2 * size**3) / (single_gpu_time / iterations) / 1e9
        
        # Test multi-GPU performance on this node
        if gpu_count > 1:
            print(f"Rank {rank}: Testing multi-GPU performance with {gpu_count} GPUs")
            
            # Split the matrix multiplication across GPUs on this node
            chunk_size = size // gpu_count
            remainder = size % gpu_count
            
            gpu_tensors = []
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                start_row = i * chunk_size
                end_row = start_row + chunk_size + (1 if i < remainder else 0)
                actual_chunk_size = end_row - start_row
                
                a_chunk = torch.randn(actual_chunk_size, size, dtype=torch.float32, device=f'cuda:{i}')
                b_full = torch.randn(size, size, dtype=torch.float32, device=f'cuda:{i}')
                gpu_tensors.append((a_chunk, b_full, actual_chunk_size))
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                results_gpu = []
                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    a_chunk, b_full, chunk_size_i = gpu_tensors[i]
                    c_chunk = torch.mm(a_chunk, b_full)
                    results_gpu.append(c_chunk)
                
                torch.cuda.synchronize()
            
            multi_gpu_time = time.time() - start_time
            results['multi_gpu_time'] = multi_gpu_time / iterations
            
            total_flops = 0
            for i in range(gpu_count):
                _, _, chunk_size_i = gpu_tensors[i]
                total_flops += 2 * chunk_size_i * size * size
            
            results['multi_gpu_gflops'] = total_flops / (multi_gpu_time / iterations) / 1e9
            results['multi_gpu_speedup'] = results['single_gpu_time'] / results['multi_gpu_time']
            
            # Clean up
            for a_chunk, b_full, _ in gpu_tensors:
                del a_chunk, b_full
            torch.cuda.empty_cache()
        
        # Gather GPU results from all nodes
        all_single_gpu_times = comm.gather(results['single_gpu_time'], root=0)
        all_single_gpu_gflops = comm.gather(results['single_gpu_gflops'], root=0)
        
        if 'multi_gpu_time' in results:
            all_multi_gpu_times = comm.gather(results['multi_gpu_time'], root=0)
            all_multi_gpu_gflops = comm.gather(results['multi_gpu_gflops'], root=0)
        else:
            all_multi_gpu_times = None
            all_multi_gpu_gflops = None
        
        # Clean up
        del a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()
    
    # Synchronize all processes
    comm.Barrier()
    
    # Root process calculates distributed performance
    if rank == 0:
        unique_nodes = len(set(all_hostnames))
        processes_per_node = size_mpi // unique_nodes
        
        print(f"\nDISTRIBUTED PERFORMANCE ANALYSIS:")
        print(f"Total nodes: {unique_nodes}")
        print(f"Processes per node: {processes_per_node}")
        print(f"Total processes: {size_mpi}")
        print(f"Total GPUs across all nodes: {unique_nodes * gpu_count}")
        
        # Calculate distributed CPU performance
        avg_cpu_time = np.mean(all_cpu_times)
        total_cpu_gflops = np.sum(all_cpu_gflops)
        print(f"Average CPU time per process: {avg_cpu_time:.4f} s")
        print(f"Total CPU GFLOPS across all processes: {total_cpu_gflops:.2f}")
        
        if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
            # Calculate distributed GPU performance
            avg_single_gpu_time = np.mean(all_single_gpu_times)
            total_single_gpu_gflops = np.sum(all_single_gpu_gflops)
            print(f"Average single GPU time per process: {avg_single_gpu_time:.4f} s")
            print(f"Total single GPU GFLOPS across all processes: {total_single_gpu_gflops:.2f}")
            
            if all_multi_gpu_times:
                avg_multi_gpu_time = np.mean(all_multi_gpu_times)
                total_multi_gpu_gflops = np.sum(all_multi_gpu_gflops)
                print(f"Average multi-GPU time per process: {avg_multi_gpu_time:.4f} s")
                print(f"Total multi-GPU GFLOPS across all processes: {total_multi_gpu_gflops:.2f}")
                
                # Calculate distributed speedup
                distributed_speedup = avg_cpu_time / avg_multi_gpu_time
                print(f"Distributed speedup (CPU vs Multi-GPU): {distributed_speedup:.2f}x")
                
                # Calculate efficiency
                if unique_nodes > 1:
                    efficiency = distributed_speedup / (unique_nodes * gpu_count)
                    print(f"Distributed efficiency: {efficiency:.2f}x per GPU")
    
    return results

def benchmark_distributed_memory_bandwidth(size_mb=2048, iterations=10):
    """Benchmark memory bandwidth using MPI across multiple nodes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_mpi = comm.Get_size()
    
    results = {}
    size = int(size_mb * 1024 * 1024 // 4)  # float32 = 4 bytes
    
    # CPU memory bandwidth (each node does its own)
    print(f"Rank {rank}: Running CPU memory bandwidth test with {size_mb}MB")
    a_cpu = np.random.random(size).astype(np.float32)
    b_cpu = np.random.random(size).astype(np.float32)
    
    # Memory copy
    start_time = time.time()
    for _ in range(iterations):
        b_cpu = a_cpu.copy()
    cpu_copy_time = time.time() - start_time
    results['cpu_copy_time'] = cpu_copy_time / iterations
    results['cpu_copy_bandwidth'] = (size_mb * iterations) / (cpu_copy_time / 1024)  # GB/s
    
    # GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        gpu_count = torch.cuda.device_count()
        print(f"Rank {rank}: Running GPU memory bandwidth test with {gpu_count} GPUs, {size_mb}MB")
        
        # Test single GPU first
        torch.cuda.set_device(0)
        a_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            b_gpu = a_gpu.clone()
        torch.cuda.synchronize()
        single_gpu_time = time.time() - start_time
        results['single_gpu_copy_time'] = single_gpu_time / iterations
        results['single_gpu_copy_bandwidth'] = (size_mb * iterations) / (single_gpu_time / 1024)  # GB/s
        
        # Test multi-GPU if we have multiple GPUs
        if gpu_count > 1:
            print(f"Rank {rank}: Testing multi-GPU memory bandwidth with {gpu_count} GPUs")
            
            # Split the memory copy across GPUs
            chunk_size = size // gpu_count
            remainder = size % gpu_count
            
            gpu_tensors = []
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                actual_chunk_size = end_idx - start_idx
                
                a_chunk = torch.randn(actual_chunk_size, dtype=torch.float32, device=f'cuda:{i}')
                b_chunk = torch.randn(actual_chunk_size, dtype=torch.float32, device=f'cuda:{i}')
                gpu_tensors.append((a_chunk, b_chunk, actual_chunk_size))
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    a_chunk, b_chunk, _ = gpu_tensors[i]
                    b_chunk = a_chunk.clone()
                
                torch.cuda.synchronize()
            
            multi_gpu_time = time.time() - start_time
            results['multi_gpu_copy_time'] = multi_gpu_time / iterations
            
            total_size_mb = 0
            for _, _, chunk_size_i in gpu_tensors:
                total_size_mb += (chunk_size_i * 4) / (1024 * 1024)  # Convert to MB
            
            results['multi_gpu_copy_bandwidth'] = (total_size_mb * iterations) / (multi_gpu_time / 1024)  # GB/s
            
            # Clean up
            for a_chunk, b_chunk, _ in gpu_tensors:
                del a_chunk, b_chunk
            torch.cuda.empty_cache()
        
        # Clean up
        del a_gpu, b_gpu
        torch.cuda.empty_cache()
    
    # Gather results from all nodes
    all_cpu_bandwidths = comm.gather(results['cpu_copy_bandwidth'], root=0)
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        all_single_gpu_bandwidths = comm.gather(results['single_gpu_copy_bandwidth'], root=0)
        if 'multi_gpu_copy_bandwidth' in results:
            all_multi_gpu_bandwidths = comm.gather(results['multi_gpu_copy_bandwidth'], root=0)
        else:
            all_multi_gpu_bandwidths = None
    else:
        all_single_gpu_bandwidths = None
        all_multi_gpu_bandwidths = None
    
    # Synchronize all processes
    comm.Barrier()
    
    # Root process calculates distributed bandwidth
    if rank == 0:
        print(f"\nDISTRIBUTED MEMORY BANDWIDTH:")
        total_cpu_bandwidth = np.sum(all_cpu_bandwidths)
        print(f"Total CPU bandwidth across all nodes: {total_cpu_bandwidth:.2f} GB/s")
        
        if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
            total_single_gpu_bandwidth = np.sum(all_single_gpu_bandwidths)
            print(f"Total single GPU bandwidth across all nodes: {total_single_gpu_bandwidth:.2f} GB/s")
            
            if all_multi_gpu_bandwidths:
                total_multi_gpu_bandwidth = np.sum(all_multi_gpu_bandwidths)
                print(f"Total multi-GPU bandwidth across all nodes: {total_multi_gpu_bandwidth:.2f} GB/s")
    
    return results

def print_distributed_results(system_info, matrix_results, memory_results):
    """Print formatted distributed results"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Gather system information from all nodes
    all_hostnames = comm.gather(system_info['hostname'], root=0)
    all_gpu_counts = comm.gather(system_info.get('gpu_count', 0), root=0)
    all_gpu_names = comm.gather(system_info.get('gpu_names', []), root=0)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED MPI BENCHMARK RESULTS")
        print(f"{'='*80}")
        
        print(f"\nSYSTEM INFORMATION:")
        print(f"  Total MPI Processes: {system_info['mpi_size']}")
        print(f"  Unique Nodes: {len(set(all_hostnames))}")
        print(f"  Processes per Node: {system_info['mpi_size'] // len(set(all_hostnames))}")
        print(f"  Node Distribution:")
        for node in sorted(set(all_hostnames)):
            count = all_hostnames.count(node)
            print(f"    {node}: {count} processes")
        print(f"  Platform: {system_info['platform']}")
        print(f"  CPU Count per Node: {system_info['cpu_count']}")
        print(f"  Memory per Node: {system_info['memory_gb']:.1f} GB")
        
        if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
            total_gpus = sum(all_gpu_counts)
            print(f"  GPU Count per Node: {all_gpu_counts}")
            print(f"  Total GPUs across all nodes: {total_gpus}")
            
            # Show GPU info from each unique node (avoid duplicates)
            unique_nodes = list(set(all_hostnames))
            for node_idx, hostname in enumerate(sorted(unique_nodes)):
                # Find the first occurrence of this hostname to get its GPU info
                first_occurrence = all_hostnames.index(hostname)
                gpu_names = all_gpu_names[first_occurrence]
                print(f"  Node {node_idx} ({hostname}):")
                for i, name in enumerate(gpu_names):
                    print(f"    GPU {i}: {name}")
        
        print(f"\n{'='*80}")

def main():
    """Main distributed benchmark function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Print MPI setup information
    print(f"Rank {rank}/{size} on {platform.node()}")
    
    if rank == 0:
        print("Starting Distributed MPI Benchmark Suite")
        print(f"This will test matrix operations and memory bandwidth across {size} nodes")
        print("Using MPI to coordinate work across multiple servers")
        print(f"MPI World Size: {size}")
        print(f"MPI Rank: {rank}")
    
    # Get system information
    system_info = get_system_info()
    
    # Run benchmarks
    if rank == 0:
        print("\nRunning distributed matrix operation benchmarks...")
    matrix_results = benchmark_distributed_matrix_operations(size=8192, iterations=3)
    
    if rank == 0:
        print("\nRunning distributed memory bandwidth benchmarks...")
    memory_results = benchmark_distributed_memory_bandwidth(size_mb=2048, iterations=10)
    
    # Print results (only on rank 0)
    print_distributed_results(system_info, matrix_results, memory_results)
    
    # Synchronize all processes
    comm.Barrier()
    
    if rank == 0:
        print("\nDistributed MPI benchmark completed successfully!")
        print("This demonstrates true multi-node GPU performance using MPI")

if __name__ == "__main__":
    main()
