# GPU Examples

This directory contains GPU benchmarking and demonstration scripts for various GPU types and configurations.

## 🚀 H100 Interactive Demo

### `h100_demo.sh` & `h100_demo.py`

**Purpose**: Interactive H100 GPU demonstration and performance showcase

**How to Run**:
```bash
# From a login node, run the interactive demo
./h100_demo.sh
```

**What it does**:
1. Launches an interactive SLURM session on the H100 partition
2. Sets up the Python environment with CUDA support
3. Runs comprehensive H100 performance demonstrations
4. Shows real-time GPU metrics and performance data

**Features**:
- **Massive Matrix Operations**: Tests matrices up to 32,768×32,768
- **Memory Bandwidth**: Demonstrates 16GB memory transfers
- **Parallel Processing**: Processes 100 million elements in parallel
- **Multi-GPU Sync**: Tests multi-GPU synchronization (if available)
- **Real-time Monitoring**: Optional live GPU utilization display

**Expected Performance**:
- Matrix operations: 10-50 TFLOPS
- Memory bandwidth: 3-4 TB/s
- Speedup vs CPU: 50-100x on large workloads

## 📊 GPU Benchmarking Scripts

### Standard Benchmarks

#### `gpu_benchmark_standard.py` & `gpu_benchmark_standard.slurm`
**Purpose**: Conservative GPU performance testing for most systems
- **Partition**: H100 (default) or L40S
- **Resources**: 1 task, 8 CPUs, 32GB memory
- **Use case**: Basic performance testing and validation

#### `gpu_benchmark_large.py` & `gpu_benchmark_large.slurm`
**Purpose**: High-performance GPU testing for H100/A100 systems
- **Partition**: H100
- **Resources**: 1 task, 16 CPUs, 128GB memory
- **Use case**: Large-scale performance testing

#### `gpu_benchmark_multi_gpu.py`
**Purpose**: Multi-GPU performance testing within a single node
- **Features**: Workload distribution across multiple GPUs
- **Scaling**: Measures actual performance improvement
- **Use case**: Multi-GPU optimization and scaling analysis

## 🎯 Usage Examples

### Interactive H100 Demo
```bash
# Launch interactive H100 demonstration
./h100_demo.sh
```

### Batch GPU Benchmarks
```bash
# Standard GPU benchmark
sbatch gpu_benchmark_standard.slurm

# Large-scale GPU benchmark
sbatch gpu_benchmark_large.slurm
```

### Multi-GPU Testing
```bash
# Multi-GPU benchmark (requires appropriate SLURM script)
python gpu_benchmark_multi_gpu.py
```

## 🔧 Requirements

### Environment Setup
- **Python**: 3.11+ with PyTorch
- **CUDA**: Compatible CUDA version for your GPU
- **SLURM**: Access to H100 or L40S partitions
- **Conda**: GPU environment with required packages

### Dependencies
- PyTorch with CUDA support
- NumPy
- psutil
- pynvml (optional, for GPU monitoring)

## 📈 Performance Expectations

### H100 Performance
- **Matrix Operations**: 10-50 TFLOPS
- **Memory Bandwidth**: 3-4 TB/s
- **Multi-GPU Scaling**: 1.5-1.8x efficiency with 2 GPUs

### L40 Performance
- **Matrix Operations**: 1-2 TFLOPS
- **Memory Bandwidth**: ~1.2 TB/s
- **Memory**: 24GB VRAM

## 🐛 Troubleshooting

### Common Issues

#### "CUDA not available"
- Check if CUDA modules are loaded
- Verify PyTorch CUDA installation
- Ensure you're on a GPU-enabled node

#### "Out of memory"
- Use smaller matrix sizes
- Check GPU memory usage with `nvidia-smi`
- Reduce batch sizes in scripts

#### "Partition not available"
- Check partition availability: `sinfo -p H100`
- Verify your access permissions
- Try alternative partitions (L40S)

### Getting Help
- Check SLURM job status: `squeue -u $USER`
- View job output: `cat <job_id>_*.out`
- Check error logs: `cat <job_id>_*.err`

## 📝 Notes

- The H100 demo is designed for interactive use to showcase GPU capabilities
- Batch scripts are available for automated testing and benchmarking
- All scripts include proper error handling and resource cleanup
- Performance metrics are logged for analysis and comparison
