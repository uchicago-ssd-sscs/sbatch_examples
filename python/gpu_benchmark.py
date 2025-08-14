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

# Try to import CuPy (but don't fail if it doesn't work)
try:
    import cupy as cp
    import cupyx
    GPU_AVAILABLE = True
    print("CuPy available - will try GPU benchmarks")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available")

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
        info['torch_gpu_name'] = torch.cuda.get_device_name(0)
        info['torch_gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_count'] = torch.cuda.device_count()
        print(f"Detected {info['gpu_count']} GPU(s): {info['torch_gpu_name']}")
    
    return info

def benchmark_matrix_operations(size=4096, iterations=10):
    """Benchmark matrix operations"""
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
    
    # Matrix addition
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = a_cpu + b_cpu
    cpu_add_time = time.time() - start_time
    results['cpu_add_time'] = cpu_add_time / iterations
    
    # GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"Running GPU matrix operations with size {size}x{size}")
        a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        # Matrix multiplication
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        results['gpu_matmul_time'] = gpu_time / iterations
        results['gpu_matmul_gflops'] = (2 * size**3) / (gpu_time / iterations) / 1e9
        
        # Matrix addition
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = a_gpu + b_gpu
        torch.cuda.synchronize()
        gpu_add_time = time.time() - start_time
        results['gpu_add_time'] = gpu_add_time / iterations
        
        # Speedup
        results['matmul_speedup'] = results['cpu_matmul_time'] / results['gpu_matmul_time']
        results['add_speedup'] = results['cpu_add_time'] / results['gpu_add_time']
    
    return results

def benchmark_memory_bandwidth(size_mb=1024, iterations=100):
    """Benchmark memory bandwidth"""
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
    
    # GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"Running GPU memory bandwidth test with {size_mb}MB")
        a_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        
        # GPU memory copy
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            b_gpu = a_gpu.clone()
        torch.cuda.synchronize()
        gpu_copy_time = time.time() - start_time
        results['gpu_copy_time'] = gpu_copy_time / iterations
        results['gpu_copy_bandwidth'] = (size_mb * iterations) / (gpu_copy_time * 1024)  # GB/s
        
        # Host to device transfer
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            a_gpu = torch.tensor(a_cpu, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        h2d_time = time.time() - start_time
        results['h2d_time'] = h2d_time / iterations
        results['h2d_bandwidth'] = (size_mb * iterations) / (h2d_time * 1024)  # GB/s
        
        # Device to host transfer
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            a_cpu = a_gpu.cpu().numpy()
        torch.cuda.synchronize()
        d2h_time = time.time() - start_time
        results['d2h_time'] = d2h_time / iterations
        results['d2h_bandwidth'] = (size_mb * iterations) / (d2h_time * 1024)  # GB/s
    
    return results

def benchmark_compute_intensive(size=1000000, iterations=1000):
    """Benchmark compute-intensive operations"""
    results = {}
    
    # CPU compute
    print(f"Running CPU compute-intensive test with {size} elements")
    a_cpu = np.random.random(size).astype(np.float32)
    b_cpu = np.random.random(size).astype(np.float32)
    
    # Element-wise operations
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = np.sin(a_cpu) * np.cos(b_cpu) + np.sqrt(np.abs(a_cpu))
    cpu_time = time.time() - start_time
    results['cpu_compute_time'] = cpu_time / iterations
    
    # GPU operations using PyTorch
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"Running GPU compute-intensive test with {size} elements")
        a_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.sin(a_gpu) * torch.cos(b_gpu) + torch.sqrt(torch.abs(a_gpu))
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        results['gpu_compute_time'] = gpu_time / iterations
        results['compute_speedup'] = results['cpu_compute_time'] / results['gpu_compute_time']
    
    return results

def benchmark_torch_operations(size=2048, iterations=10):
    """Benchmark PyTorch operations if available"""
    results = {}
    
    if not TORCH_AVAILABLE:
        return results
    
    print(f"Running PyTorch operations with size {size}x{size}")
    
    # CPU PyTorch
    a_cpu = torch.randn(size, size, dtype=torch.float32)
    b_cpu = torch.randn(size, size, dtype=torch.float32)
    
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = torch.mm(a_cpu, b_cpu)
    torch.cuda.synchronize() if TORCH_GPU_AVAILABLE else None
    cpu_time = time.time() - start_time
    results['torch_cpu_time'] = cpu_time / iterations
    
    if TORCH_GPU_AVAILABLE:
        a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        results['torch_gpu_time'] = gpu_time / iterations
        results['torch_speedup'] = results['torch_cpu_time'] / results['torch_gpu_time']
    
    return results

def print_results(system_info, matrix_results, memory_results, compute_results, torch_results):
    """Print formatted results"""
    print(f"\n{'='*80}")
    print(f"GPU/CPU BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print(f"\nSYSTEM INFORMATION:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  Platform: {system_info['platform']}")
    print(f"  CPU Count: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory_gb']:.1f} GB")
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  GPU: {system_info['torch_gpu_name']}")
        print(f"  GPU Memory: {system_info['torch_gpu_memory_gb']:.1f} GB")
    
    print(f"\nMATRIX OPERATIONS:")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_time', 0):.4f} s")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_gflops', 0):.2f} GFLOPS")
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  GPU Matrix Multiplication: {matrix_results.get('gpu_matmul_time', 0):.4f} s")
        print(f"  GPU Matrix Multiplication: {matrix_results.get('gpu_matmul_gflops', 0):.2f} GFLOPS")
        print(f"  Matrix Multiplication Speedup: {matrix_results.get('matmul_speedup', 0):.2f}x")
    
    print(f"\nMEMORY BANDWIDTH:")
    print(f"  CPU Copy Bandwidth: {memory_results.get('cpu_copy_bandwidth', 0):.2f} GB/s")
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  GPU Copy Bandwidth: {memory_results.get('gpu_copy_bandwidth', 0):.2f} GB/s")
        print(f"  H2D Bandwidth: {memory_results.get('h2d_bandwidth', 0):.2f} GB/s")
        print(f"  D2H Bandwidth: {memory_results.get('d2h_bandwidth', 0):.2f} GB/s")
    
    print(f"\nCOMPUTE INTENSIVE:")
    print(f"  CPU Compute Time: {compute_results.get('cpu_compute_time', 0):.4f} s")
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  GPU Compute Time: {compute_results.get('gpu_compute_time', 0):.4f} s")
        print(f"  Compute Speedup: {compute_results.get('compute_speedup', 0):.2f}x")
    
    if TORCH_AVAILABLE:
        print(f"\nPYTORCH OPERATIONS:")
        print(f"  PyTorch CPU Time: {torch_results.get('torch_cpu_time', 0):.4f} s")
        if TORCH_GPU_AVAILABLE:
            print(f"  PyTorch GPU Time: {torch_results.get('torch_gpu_time', 0):.4f} s")
            print(f"  PyTorch Speedup: {torch_results.get('torch_speedup', 0):.2f}x")
    
    print(f"\n{'='*80}")

def main():
    """Main benchmark function"""
    print("Starting GPU/CPU Benchmark Suite")
    print("This will test matrix operations, memory bandwidth, and compute performance")
    print("Comparing CPU vs GPU performance using PyTorch")
    
    # Get system information
    system_info = get_system_info()
    
    # Run benchmarks with larger matrices for better GPU demonstration
    print("\nRunning matrix operation benchmarks...")
    matrix_results = benchmark_matrix_operations(size=8192, iterations=3)
    
    print("\nRunning memory bandwidth benchmarks...")
    memory_results = benchmark_memory_bandwidth(size_mb=2048, iterations=20)
    
    print("\nRunning compute-intensive benchmarks...")
    compute_results = benchmark_compute_intensive(size=2000000, iterations=200)
    
    print("\nRunning PyTorch benchmarks...")
    torch_results = benchmark_torch_operations(size=4096, iterations=3)
    
    # Print results
    print_results(system_info, matrix_results, memory_results, compute_results, torch_results)
    
    print("\nBenchmark completed successfully!")
    print("Use these results to compare CPU vs GPU performance")

if __name__ == "__main__":
    main()
