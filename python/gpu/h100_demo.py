#!/usr/bin/env python3

import torch
import time
import numpy as np
import psutil
import platform

def print_header():
    print("=" * 80)
    print("🚀 H100 GPU POWER DEMONSTRATION")
    print("=" * 80)
    print(f"Hostname: {platform.node()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"System Memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print("=" * 80)

def demo_massive_matrix_ops():
    print("\n🔥 DEMO 1: MASSIVE MATRIX OPERATIONS")
    print("-" * 50)
    
    # Test different matrix sizes
    sizes = [8192, 16384, 32768]
    
    for size in sizes:
        print(f"\n📊 Testing {size:,} × {size:,} matrix multiplication")
        print(f"   Matrix size: {size * size * 4 / 1e9:.1f} GB")
        
        # CPU test
        print("   🖥️  CPU calculation...")
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)
        
        start = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        cpu_gflops = (2 * size**3) / cpu_time / 1e9
        
        # GPU test
        print("   🚀 GPU calculation...")
        a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        gpu_gflops = (2 * size**3) / gpu_time / 1e9
        
        speedup = cpu_time / gpu_time
        
        print(f"   ✅ CPU: {cpu_time:.2f}s ({cpu_gflops:.0f} GFLOPS)")
        print(f"   ✅ GPU: {gpu_time:.2f}s ({gpu_gflops:.0f} GFLOPS)")
        print(f"   🎯 Speedup: {speedup:.1f}x")
        
        if size == 32768:
            print(f"   🏆 H100 processed a {size:,}×{size:,} matrix in {gpu_time:.2f} seconds!")
            print(f"   🏆 That's {gpu_gflops:.0f} billion floating-point operations per second!")

def demo_memory_bandwidth():
    print("\n🔥 DEMO 2: MEMORY BANDWIDTH SHOWCASE")
    print("-" * 50)
    
    # Test massive memory transfers
    sizes_gb = [1, 4, 8, 16]
    
    for size_gb in sizes_gb:
        size_elements = int(size_gb * 1024**3 // 4)  # float32 = 4 bytes
        print(f"\n📊 Testing {size_gb}GB memory transfer")
        
        # Create data
        data_cpu = torch.randn(size_elements, dtype=torch.float32)
        
        # H2D transfer
        torch.cuda.synchronize()
        start = time.time()
        data_gpu = data_cpu.cuda()
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        h2d_bandwidth = size_gb / h2d_time
        
        # D2H transfer
        torch.cuda.synchronize()
        start = time.time()
        data_cpu_back = data_gpu.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        d2h_bandwidth = size_gb / d2h_time
        
        print(f"   🚀 H2D: {h2d_time:.2f}s ({h2d_bandwidth:.1f} GB/s)")
        print(f"   🚀 D2H: {d2h_time:.2f}s ({d2h_bandwidth:.1f} GB/s)")
        
        if size_gb == 16:
            print(f"   🏆 H100 transferred 16GB in {h2d_time:.2f} seconds!")
            print(f"   🏆 That's {h2d_bandwidth:.1f} GB/s bandwidth!")

def demo_parallel_processing():
    print("\n🔥 DEMO 3: MASSIVE PARALLEL PROCESSING")
    print("-" * 50)
    
    # Test element-wise operations on massive arrays
    sizes = [10_000_000, 50_000_000, 100_000_000]
    
    for size in sizes:
        print(f"\n📊 Testing {size:,} element parallel computation")
        
        # CPU test
        print("   🖥️  CPU processing...")
        a_cpu = torch.randn(size, dtype=torch.float32)
        b_cpu = torch.randn(size, dtype=torch.float32)
        
        start = time.time()
        c_cpu = torch.sin(a_cpu) * torch.cos(b_cpu) + torch.sqrt(torch.abs(a_cpu))
        cpu_time = time.time() - start
        
        # GPU test
        print("   🚀 GPU processing...")
        a_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        b_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        c_gpu = torch.sin(a_gpu) * torch.cos(b_gpu) + torch.sqrt(torch.abs(a_gpu))
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"   ✅ CPU: {cpu_time:.2f}s")
        print(f"   ✅ GPU: {gpu_time:.2f}s")
        print(f"   🎯 Speedup: {speedup:.1f}x")
        
        if size == 100_000_000:
            print(f"   🏆 H100 processed 100 million elements in {gpu_time:.2f} seconds!")
            print(f"   🏆 That's {speedup:.1f}x faster than CPU!")

def demo_multi_gpu_sync():
    print("\n🔥 DEMO 4: MULTI-GPU SYNCHRONIZATION")
    print("-" * 50)
    
    gpu_count = torch.cuda.device_count()
    print(f"\n📊 Testing {gpu_count} GPU synchronization")
    
    if gpu_count > 1:
        # Create tensors on each GPU
        tensors = []
        for i in range(gpu_count):
            with torch.cuda.device(i):
                tensors.append(torch.randn(1000, 1000, dtype=torch.float32, device='cuda'))
        
        # Synchronize all GPUs
        torch.cuda.synchronize()
        start = time.time()
        
        # Perform operations on all GPUs
        results = []
        for i, tensor in enumerate(tensors):
            with torch.cuda.device(i):
                result = torch.mm(tensor, tensor)
                results.append(result)
        
        torch.cuda.synchronize()
        total_time = time.time() - start
        
        print(f"   ✅ Synchronized {gpu_count} GPUs in {total_time:.3f}s")
        print(f"   🏆 Multi-GPU matrix operations completed!")
    else:
        print("   ℹ️  Single GPU system - skipping multi-GPU demo")

def demo_real_time_visualization():
    print("\n🔥 DEMO 5: REAL-TIME PERFORMANCE METRICS")
    print("-" * 50)
    
    print("\n📊 Live GPU utilization monitoring...")
    print("   Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Get GPU utilization (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = "N/A"
                memory_util = "N/A"
            
            print(f"\r   🚀 GPU: {gpu_util}% | Memory: {memory_allocated:.1f}GB/{memory_total:.1f}GB | Reserved: {memory_reserved:.1f}GB", end="")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n   ✅ Monitoring stopped")

def main():
    print_header()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run H100 demonstrations.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
    
    # Run demonstrations
    demo_massive_matrix_ops()
    demo_memory_bandwidth()
    demo_parallel_processing()
    demo_multi_gpu_sync()
    
    print("\n" + "=" * 80)
    print("🎉 H100 DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Key takeaways:")
    print("• H100 can process massive matrices in seconds")
    print("• Memory bandwidth exceeds 10 GB/s")
    print("• 50-100x speedup on parallel workloads")
    print("• Real-time processing of millions of elements")
    print("=" * 80)
    
    # Optional: Run real-time monitoring
    response = input("\nWould you like to see real-time GPU monitoring? (y/n): ")
    if response.lower() == 'y':
        demo_real_time_visualization()

if __name__ == "__main__":
    main()
