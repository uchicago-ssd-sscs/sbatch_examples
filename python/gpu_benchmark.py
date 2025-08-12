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
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available - running single-process benchmarks")
    # Create a mock MPI-like object for single-process execution
    class MockMPI:
        class COMM_WORLD:
            @staticmethod
            def Get_rank():
                return 0
            @staticmethod
            def Get_size():
                return 1
            @staticmethod
            def Barrier():
                pass
    MPI = MockMPI()

# Try to import GPU libraries
try:
    import cupy as cp
    import cupyx
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - running CPU-only benchmarks")

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_GPU_AVAILABLE = True
    else:
        TORCH_GPU_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_GPU_AVAILABLE = False

def get_system_info():
    """Get detailed system information"""
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'mpi_rank': MPI.COMM_WORLD.Get_rank(),
        'mpi_size': MPI.COMM_WORLD.Get_size()
    }
    
    if GPU_AVAILABLE:
        try:
            info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
            info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            info['gpu_memory_gb'] = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        except:
            info['gpu_count'] = 0
            info['gpu_name'] = 'Unknown'
            info['gpu_memory_gb'] = 0
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        info['torch_gpu_name'] = torch.cuda.get_device_name(0)
        info['torch_gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
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
    
    if GPU_AVAILABLE:
        print(f"Running GPU matrix operations with size {size}x{size}")
        a_gpu = cp.random.random((size, size)).astype(cp.float32)
        b_gpu = cp.random.random((size, size)).astype(cp.float32)
        
        # Synchronize before timing
        cp.cuda.Stream.null.synchronize()
        
        # Matrix multiplication
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        results['gpu_matmul_time'] = gpu_time / iterations
        results['gpu_matmul_gflops'] = (2 * size**3) / (gpu_time / iterations) / 1e9
        
        # Matrix addition
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()
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
    results['cpu_copy_bandwidth'] = (size_mb * iterations) / (cpu_copy_time / 1024)  # GB/s
    
    if GPU_AVAILABLE:
        print(f"Running GPU memory bandwidth test with {size_mb}MB")
        a_gpu = cp.random.random(size).astype(cp.float32)
        b_gpu = cp.random.random(size).astype(cp.float32)
        
        # GPU memory copy
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            b_gpu = a_gpu.copy()
        cp.cuda.Stream.null.synchronize()
        gpu_copy_time = time.time() - start_time
        results['gpu_copy_time'] = gpu_copy_time / iterations
        results['gpu_copy_bandwidth'] = (size_mb * iterations) / (gpu_copy_time / 1024)  # GB/s
        
        # Host to device transfer
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            a_gpu = cp.asarray(a_cpu)
        cp.cuda.Stream.null.synchronize()
        h2d_time = time.time() - start_time
        results['h2d_time'] = h2d_time / iterations
        results['h2d_bandwidth'] = (size_mb * iterations) / (h2d_time / 1024)  # GB/s
        
        # Device to host transfer
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            a_cpu = cp.asnumpy(a_gpu)
        cp.cuda.Stream.null.synchronize()
        d2h_time = time.time() - start_time
        results['d2h_time'] = d2h_time / iterations
        results['d2h_bandwidth'] = (size_mb * iterations) / (d2h_time / 1024)  # GB/s
    
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
    
    if GPU_AVAILABLE:
        print(f"Running GPU compute-intensive test with {size} elements")
        a_gpu = cp.random.random(size).astype(cp.float32)
        b_gpu = cp.random.random(size).astype(cp.float32)
        
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = cp.sin(a_gpu) * cp.cos(b_gpu) + cp.sqrt(cp.abs(a_gpu))
        cp.cuda.Stream.null.synchronize()
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
    rank = system_info['mpi_rank']
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - Rank {rank}")
    print(f"{'='*80}")
    
    print(f"\nSYSTEM INFORMATION:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  Platform: {system_info['platform']}")
    print(f"  CPU Count: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory_gb']:.1f} GB")
    print(f"  MPI Rank: {system_info['mpi_rank']}/{system_info['mpi_size']}")
    
    if GPU_AVAILABLE:
        print(f"  GPU: {system_info['gpu_name']}")
        print(f"  GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")
    
    if TORCH_AVAILABLE and TORCH_GPU_AVAILABLE:
        print(f"  PyTorch GPU: {system_info['torch_gpu_name']}")
        print(f"  PyTorch GPU Memory: {system_info['torch_gpu_memory_gb']:.1f} GB")
    
    print(f"\nMATRIX OPERATIONS:")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_time', 0):.4f} s")
    print(f"  CPU Matrix Multiplication: {matrix_results.get('cpu_matmul_gflops', 0):.2f} GFLOPS")
    if GPU_AVAILABLE:
        print(f"  GPU Matrix Multiplication: {matrix_results.get('gpu_matmul_time', 0):.4f} s")
        print(f"  GPU Matrix Multiplication: {matrix_results.get('gpu_matmul_gflops', 0):.2f} GFLOPS")
        print(f"  Matrix Multiplication Speedup: {matrix_results.get('matmul_speedup', 0):.2f}x")
    
    print(f"\nMEMORY BANDWIDTH:")
    print(f"  CPU Copy Bandwidth: {memory_results.get('cpu_copy_bandwidth', 0):.2f} GB/s")
    if GPU_AVAILABLE:
        print(f"  GPU Copy Bandwidth: {memory_results.get('gpu_copy_bandwidth', 0):.2f} GB/s")
        print(f"  H2D Bandwidth: {memory_results.get('h2d_bandwidth', 0):.2f} GB/s")
        print(f"  D2H Bandwidth: {memory_results.get('d2h_bandwidth', 0):.2f} GB/s")
    
    print(f"\nCOMPUTE INTENSIVE:")
    print(f"  CPU Compute Time: {compute_results.get('cpu_compute_time', 0):.4f} s")
    if GPU_AVAILABLE:
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Starting GPU/CPU Benchmark Suite")
        print("This will test matrix operations, memory bandwidth, and compute performance")
        if MPI_AVAILABLE:
            print(f"Running with MPI - {comm.Get_size()} processes")
        else:
            print("Running in single-process mode")
        print("Gathering CPU and GPU performance for comparison")
    
    # Get system information
    system_info = get_system_info()
    
    # Run benchmarks
    if rank == 0:
        print("\nRunning matrix operation benchmarks...")
    matrix_results = benchmark_matrix_operations(size=4096, iterations=5)
    
    if rank == 0:
        print("\nRunning memory bandwidth benchmarks...")
    memory_results = benchmark_memory_bandwidth(size_mb=512, iterations=50)
    
    if rank == 0:
        print("\nRunning compute-intensive benchmarks...")
    compute_results = benchmark_compute_intensive(size=500000, iterations=500)
    
    if rank == 0:
        print("\nRunning PyTorch benchmarks...")
    torch_results = benchmark_torch_operations(size=2048, iterations=5)
    
    # Print results
    print_results(system_info, matrix_results, memory_results, compute_results, torch_results)
    
    # Synchronize all processes (only if MPI is available)
    if MPI_AVAILABLE:
        comm.Barrier()
    
    if rank == 0:
        print("\nBenchmark completed successfully!")
        print("Use these results to compare H100 vs L40 performance")

if __name__ == "__main__":
    main() 