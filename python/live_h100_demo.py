#!/usr/bin/env python3

import torch
import time
import numpy as np
import psutil
import platform

def print_banner():
    print("=" * 80)
    print("🚀 LIVE H100 GPU POWER DEMONSTRATION")
    print("=" * 80)
    print(f"Hostname: {platform.node()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print("=" * 80)

def demo_1_matrix_speedup():
    print("\n🔥 DEMO 1: Matrix Multiplication Speedup")
    print("-" * 50)
    
    size = 8192
    print(f"📊 Testing {size:,} × {size:,} matrix multiplication")
    print(f"   Matrix size: {size * size * 4 / 1e9:.1f} GB")
    
    # CPU test
    print("\n   🖥️  CPU calculation (this will take a while)...")
    a_cpu = torch.randn(size, size, dtype=torch.float32)
    b_cpu = torch.randn(size, size, dtype=torch.float32)
    
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start
    cpu_gflops = (2 * size**3) / cpu_time / 1e9
    
    print(f"   ✅ CPU completed in {cpu_time:.2f} seconds ({cpu_gflops:.0f} GFLOPS)")
    
    # GPU test
    print("\n   🚀 GPU calculation (watch how fast this is)...")
    a_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
    b_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    gpu_gflops = (2 * size**3) / gpu_time / 1e9
    
    speedup = cpu_time / gpu_time
    
    print(f"   ✅ GPU completed in {gpu_time:.2f} seconds ({gpu_gflops:.0f} GFLOPS)")
    print(f"   🎯 H100 is {speedup:.1f}x faster than CPU!")
    print(f"   🏆 That's {gpu_gflops:.0f} billion operations per second!")

def demo_2_memory_bandwidth():
    print("\n🔥 DEMO 2: Memory Bandwidth Showcase")
    print("-" * 50)
    
    size_gb = 4
    size_elements = int(size_gb * 1024**3 // 4)
    print(f"📊 Testing {size_gb}GB memory transfer")
    
    # Create data
    data_cpu = torch.randn(size_elements, dtype=torch.float32)
    
    # H2D transfer
    print("\n   🚀 Transferring {size_gb}GB from CPU to GPU...")
    torch.cuda.synchronize()
    start = time.time()
    data_gpu = data_cpu.cuda()
    torch.cuda.synchronize()
    h2d_time = time.time() - start
    h2d_bandwidth = size_gb / h2d_time
    
    print(f"   ✅ H2D transfer: {h2d_time:.2f}s ({h2d_bandwidth:.1f} GB/s)")
    
    # D2H transfer
    print("\n   🚀 Transferring {size_gb}GB from GPU to CPU...")
    torch.cuda.synchronize()
    start = time.time()
    data_cpu_back = data_gpu.cpu()
    torch.cuda.synchronize()
    d2h_time = time.time() - start
    d2h_bandwidth = size_gb / d2h_time
    
    print(f"   ✅ D2H transfer: {d2h_time:.2f}s ({d2h_bandwidth:.1f} GB/s)")
    print(f"   🏆 H100 memory bandwidth: {h2d_bandwidth:.1f} GB/s!")

def demo_3_parallel_processing():
    print("\n🔥 DEMO 3: Massive Parallel Processing")
    print("-" * 50)
    
    size = 50_000_000
    print(f"📊 Testing {size:,} element parallel computation")
    print("   (Complex mathematical operations on 50 million elements)")
    
    # CPU test
    print("\n   🖥️  CPU processing (this will take a while)...")
    a_cpu = torch.randn(size, dtype=torch.float32)
    b_cpu = torch.randn(size, dtype=torch.float32)
    
    start = time.time()
    c_cpu = torch.sin(a_cpu) * torch.cos(b_cpu) + torch.sqrt(torch.abs(a_cpu))
    cpu_time = time.time() - start
    
    print(f"   ✅ CPU completed in {cpu_time:.2f} seconds")
    
    # GPU test
    print("\n   🚀 GPU processing (watch the speed difference)...")
    a_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
    b_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = torch.sin(a_gpu) * torch.cos(b_gpu) + torch.sqrt(torch.abs(a_gpu))
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    print(f"   ✅ GPU completed in {gpu_time:.2f} seconds")
    print(f"   🎯 H100 is {speedup:.1f}x faster than CPU!")
    print(f"   🏆 Processed 50 million elements in {gpu_time:.2f} seconds!")

def demo_4_live_monitoring():
    print("\n🔥 DEMO 4: Live GPU Monitoring")
    print("-" * 50)
    
    print("📊 Live GPU utilization and memory usage")
    print("   Press Ctrl+C to stop monitoring")
    print("")
    
    try:
        for i in range(10):  # Show for 10 seconds
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

def interactive_menu():
    print("\n" + "=" * 80)
    print("🎯 INTERACTIVE DEMO MENU")
    print("=" * 80)
    print("1. Matrix Multiplication Speedup")
    print("2. Memory Bandwidth Showcase")
    print("3. Massive Parallel Processing")
    print("4. Live GPU Monitoring")
    print("5. Run All Demos")
    print("6. Exit")
    print("=" * 80)
    
    while True:
        try:
            choice = input("\nSelect demo (1-6): ").strip()
            
            if choice == '1':
                demo_1_matrix_speedup()
            elif choice == '2':
                demo_2_memory_bandwidth()
            elif choice == '3':
                demo_3_parallel_processing()
            elif choice == '4':
                demo_4_live_monitoring()
            elif choice == '5':
                demo_1_matrix_speedup()
                demo_2_memory_bandwidth()
                demo_3_parallel_processing()
                demo_4_live_monitoring()
            elif choice == '6':
                print("\n🎉 Demo completed! Thanks for watching!")
                break
            else:
                print("Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n🎉 Demo interrupted. Thanks for watching!")
            break

def main():
    print_banner()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot run H100 demonstrations.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
    
    # Check if running interactively
    if input("\nRun interactive menu? (y/n): ").lower() == 'y':
        interactive_menu()
    else:
        # Run all demos automatically
        demo_1_matrix_speedup()
        demo_2_memory_bandwidth()
        demo_3_parallel_processing()
        
        print("\n" + "=" * 80)
        print("🎉 H100 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("Key takeaways:")
        print("• H100 can process massive matrices in seconds")
        print("• Memory bandwidth exceeds 10 GB/s")
        print("• 50-100x speedup on parallel workloads")
        print("• Real-time processing of millions of elements")
        print("=" * 80)

if __name__ == "__main__":
    main()
