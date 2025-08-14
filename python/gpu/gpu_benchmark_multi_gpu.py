#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
import platform
import psutil

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
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
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
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return info

def benchmark_multi_gpu_matrix_operations(size=8192, iterations=3):
    """Benchmark matrix operations using multiple GPUs with proper workload splitting"""
    results = {}
    
    # CPU matrix operations
    print(f"Running CPU matrix operations with size {size}x{size}")
    a_cpu = np.random.random((size, size)).astype(np.float32)
    b_cpu = np.random.random((size, size)).astype(np.float32)
    
    # Matrix multiplication
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    results['cpu_matmul_time'] = cpu_time / iterations
    results['cpu_matmul_gflops'] = (2 * size**3) / (cpu_time / iterations) / 1e9
    
    # Multi-GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        gpu_count = torch.cuda.device_count()
        print(f"Running multi-GPU matrix operations with {gpu_count} GPUs, size {size}x{size}")
        
        # Test single GPU performance first
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
        
        # Test multi-GPU performance if we have multiple GPUs
        if gpu_count > 1:
            print(f"Testing multi-GPU performance with {gpu_count} GPUs (splitting workload)...")
            
            # Split the matrix multiplication across GPUs
            # We'll split matrix A into chunks and distribute them across GPUs
            chunk_size = size // gpu_count
            remainder = size % gpu_count
            
            # Create tensors on each GPU
            gpu_tensors = []
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                # Calculate chunk size for this GPU
                start_row = i * chunk_size
                end_row = start_row + chunk_size + (1 if i < remainder else 0)
                actual_chunk_size = end_row - start_row
                
                # Create chunk of matrix A for this GPU
                a_chunk = torch.randn(actual_chunk_size, size, dtype=torch.float32, device=f'cuda:{i}')
                # Full matrix B on each GPU
                b_full = torch.randn(size, size, dtype=torch.float32, device=f'cuda:{i}')
                gpu_tensors.append((a_chunk, b_full, actual_chunk_size))
            
            # Synchronize all GPUs
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                # Run matrix multiplication on each GPU in parallel
                results_gpu = []
                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    a_chunk, b_full, chunk_size_i = gpu_tensors[i]
                    c_chunk = torch.mm(a_chunk, b_full)  # This is the actual work
                    results_gpu.append(c_chunk)
                
                # Synchronize all GPUs
                torch.cuda.synchronize()
            
            multi_gpu_time = time.time() - start_time
            results['multi_gpu_time'] = multi_gpu_time / iterations
            
            # Calculate total FLOPS: each GPU does (2 * chunk_size * size * size) operations
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
        
        # Speedup vs CPU
        results['cpu_gpu_speedup'] = results['cpu_matmul_time'] / results['single_gpu_time']
        
        # Clean up
        del a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()
    
    return results

def benchmark_multi_gpu_memory_bandwidth(size_mb=2048, iterations=10):
    """Benchmark memory bandwidth using multiple GPUs with proper workload splitting"""
    results = {}
    size = int(size_mb * 1024 * 1024 // 4)  # float32 = 4 bytes
    
    # CPU memory bandwidth
    print(f"Running CPU memory bandwidth test with {size_mb}MB")
    a_cpu = np.random.random(size).astype(np.float32)
    b_cpu = np.random.random(size).astype(np.float32)
    
    # Memory copy
    start_time = time.time()
    for _ in range(iterations):
        b_cpu = a_cpu.copy()
    cpu_copy_time = time.time() - start_time
    results['cpu_copy_time'] = cpu_copy_time / iterations
    results['cpu_copy_bandwidth'] = (size_mb * iterations) / (cpu_copy_time * 1024)  # GB/s
    
    # Multi-GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        gpu_count = torch.cuda.device_count()
        print(f"Running multi-GPU memory bandwidth test with {gpu_count} GPUs, {size_mb}MB")
        
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
        results['single_gpu_copy_bandwidth'] = (size_mb * iterations) / (single_gpu_time * 1024)  # GB/s
        
        # Test multi-GPU if we have multiple GPUs
        if gpu_count > 1:
            print(f"Testing multi-GPU memory bandwidth with {gpu_count} GPUs (splitting workload)...")
            
            # Split the memory copy across GPUs
            chunk_size = size // gpu_count
            remainder = size % gpu_count
            
            # Create tensors on each GPU
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
                # Run memory copy on each GPU in parallel
                for i in range(gpu_count):
                    torch.cuda.set_device(i)
                    a_chunk, b_chunk, _ = gpu_tensors[i]
                    b_chunk = a_chunk.clone()
                
                torch.cuda.synchronize()
            
            multi_gpu_time = time.time() - start_time
            results['multi_gpu_copy_time'] = multi_gpu_time / iterations
            
            # Calculate total bandwidth: sum of all chunks
            total_size_mb = 0
            for _, _, chunk_size_i in gpu_tensors:
                total_size_mb += (chunk_size_i * 4) / (1024 * 1024)  # Convert to MB
            
            results['multi_gpu_copy_bandwidth'] = (total_size_mb * iterations) / (multi_gpu_time * 1024)  # GB/s
            
            # Clean up
            for a_chunk, b_chunk, _ in gpu_tensors:
                del a_chunk, b_chunk
            torch.cuda.empty_cache()
        
        # Clean up
        del a_gpu, b_gpu
        torch.cuda.empty_cache()
    
    return results

def print_multi_gpu_results(system_info, matrix_results, memory_results):
    """Print formatted multi-GPU results"""
    print(f"\n{'='*80}")
    print(f"MULTI-GPU BENCHMARK RESULTS (CORRECTED)")
    print(f"{'='*80}")
    
    print(f"\nSYSTEM INFORMATION:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  Platform: {system_info['platform']}")
    print(f"  CPU Count: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory_gb']:.1f} GB")
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  GPU Count: {system_info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(system_info['gpu_names'], system_info['gpu_memories'])):
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    
    print(f"\nMATRIX OPERATIONS:")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_time', 0):.4f} s")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_gflops', 0):.2f} GFLOPS")
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  Single GPU Time: {matrix_results.get('single_gpu_time', 0):.4f} s")
        print(f"  Single GPU GFLOPS: {matrix_results.get('single_gpu_gflops', 0):.2f} GFLOPS")
        print(f"  CPU vs Single GPU Speedup: {matrix_results.get('cpu_gpu_speedup', 0):.2f}x")
        
        if 'multi_gpu_time' in matrix_results:
            print(f"  Multi-GPU Time: {matrix_results.get('multi_gpu_time', 0):.4f} s")
            print(f"  Multi-GPU GFLOPS: {matrix_results.get('multi_gpu_gflops', 0):.2f} GFLOPS")
            print(f"  Multi-GPU vs Single GPU Speedup: {matrix_results.get('multi_gpu_speedup', 0):.2f}x")
            print(f"  Multi-GPU Efficiency: {(matrix_results.get('multi_gpu_speedup', 0) / system_info['gpu_count'] * 100):.1f}%")
    
    print(f"\nMEMORY BANDWIDTH:")
    print(f"  CPU Copy Bandwidth: {memory_results.get('cpu_copy_bandwidth', 0):.2f} GB/s")
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  Single GPU Copy Bandwidth: {memory_results.get('single_gpu_copy_bandwidth', 0):.2f} GB/s")
        if 'multi_gpu_copy_bandwidth' in memory_results:
            print(f"  Multi-GPU Copy Bandwidth: {memory_results.get('multi_gpu_copy_bandwidth', 0):.2f} GB/s")
    
    print(f"\n{'='*80}")

def main():
    """Main multi-GPU benchmark function"""
    print("Starting Multi-GPU Benchmark Suite (Corrected)")
    print("This will test matrix operations and memory bandwidth across multiple GPUs")
    print("Comparing single GPU vs multi-GPU performance with proper workload splitting")
    
    # Get system information
    system_info = get_system_info()
    
    # Run benchmarks
    print("\nRunning multi-GPU matrix operation benchmarks...")
    matrix_results = benchmark_multi_gpu_matrix_operations(size=8192, iterations=3)
    
    print("\nRunning multi-GPU memory bandwidth benchmarks...")
    memory_results = benchmark_multi_gpu_memory_bandwidth(size_mb=2048, iterations=10)
    
    # Print results
    print_multi_gpu_results(system_info, matrix_results, memory_results)
    
    print("\nMulti-GPU benchmark completed successfully!")
    print("These results now show realistic multi-GPU performance with proper workload distribution")

if __name__ == "__main__":
    main()
