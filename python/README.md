# GPU Benchmarking Suite

This directory contains a comprehensive GPU benchmarking suite designed to test CPU vs GPU performance across single nodes and multi-node clusters using PyTorch and MPI.

## 🚀 Quick Start

### 1. Install Miniconda

First, install Miniconda in your home directory:

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda (accept defaults for home directory installation)
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell or source bashrc
source ~/.bashrc
```

### 2. Create Conda Environment

Create the GPU benchmarking environment from the provided `environment.yml`:

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate gpu
```

### 3. Run Benchmarks

Choose the appropriate benchmark for your needs:

```bash
# Single GPU benchmark
sbatch gpu_benchmark.slurm

# Multi-GPU benchmark (single node)
sbatch gpu_benchmark_multi_gpu.slurm

# Multi-node MPI benchmark
sbatch gpu_benchmark_multi.slurm

# Large-scale benchmark (optimized for L40)
sbatch gpu_benchmark_l40.slurm
```

## 📋 Script Descriptions

### Core Benchmark Scripts

#### `gpu_benchmark.py`
**Purpose**: Basic GPU/CPU performance comparison on a single node
- **Matrix operations**: CPU vs GPU matrix multiplication and addition
- **Memory bandwidth**: CPU copy, GPU copy, host-to-device, device-to-host transfers
- **Compute-intensive**: Element-wise mathematical operations
- **PyTorch operations**: CPU vs GPU matrix multiplication with PyTorch
- **No MPI required**: Runs on single node with single or multiple GPUs

#### `gpu_benchmark_multi_gpu.py`
**Purpose**: Multi-GPU performance testing within a single node
- **Workload splitting**: Properly distributes work across multiple GPUs on one node
- **Realistic scaling**: Measures actual performance improvement, not false parallelism
- **Memory management**: Handles GPU memory efficiently with cleanup
- **Efficiency metrics**: Shows how well multi-GPU scaling works

#### `gpu_benchmark_mpi_multi_node.py`
**Purpose**: Distributed GPU benchmarking across multiple nodes using MPI
- **MPI coordination**: Uses MPI to coordinate work across multiple servers
- **Node aggregation**: Gathers results from all nodes and calculates total performance
- **Distributed analysis**: Shows performance across entire cluster
- **Multi-node scaling**: Demonstrates true distributed GPU performance

#### `gpu_benchmark_l40.py`
**Purpose**: Large-scale benchmarking optimized for L40 hardware
- **Memory management**: Automatically handles GPU memory limits
- **Fallback mechanisms**: Reduces workload if GPU memory is insufficient
- **L40 optimization**: Designed for ~24GB VRAM (L40 has 24GB)
- **Larger workloads**: Uses bigger matrices to better demonstrate GPU advantages

### SLURM Submission Scripts

#### `gpu_benchmark.slurm`
- **Partition**: H100 (default) or L40S
- **Resources**: 1 task, 8 CPUs, 32GB memory
- **GPU**: Uses available GPUs on single node
- **Use case**: Basic performance testing

#### `gpu_benchmark_multi_gpu.slurm`
- **Partition**: L40S
- **Resources**: 1 task, 8 CPUs, 64GB memory
- **GPU**: Multiple GPUs on single node
- **Use case**: Multi-GPU performance testing

#### `gpu_benchmark_multi.slurm`
- **Partition**: L40S
- **Resources**: 2 nodes, 1 task per node, 64GB memory per node
- **GPU**: Multiple GPUs across multiple nodes
- **Network test**: Includes cross-node network bandwidth testing
- **Use case**: Multi-node distributed GPU performance + network testing
- **Resources**: 2 tasks, 2 nodes, 8 CPUs per task, 64GB memory
- **GPU**: Multiple GPUs across multiple nodes
- **MPI**: Uses `srun --mpi=pmi2` for proper multi-node coordination
- **Use case**: Distributed multi-node GPU performance testing

#### `gpu_benchmark_l40.slurm`
- **Partition**: L40S
- **Resources**: 1 task, 8 CPUs, 64GB memory
- **GPU**: Optimized for L40 memory constraints
- **Use case**: Large-scale benchmarking on L40 hardware

### Testing and Debugging Scripts

#### `mpi_test.py` & `mpi_test.slurm`
**Purpose**: Verify MPI multi-node communication is working
- **Simple test**: Basic MPI rank and hostname verification
- **Communication test**: Message passing between nodes
- **Network test**: Cross-node network bandwidth measurement
- **Diagnostic tool**: Helps troubleshoot MPI setup issues

#### `network_bandwidth_test.py`
**Purpose**: Measure actual network bandwidth between nodes
- **Point-to-point**: Tests data transfer between specific nodes
- **Bidirectional**: Measures simultaneous send/receive performance
- **All-to-all**: Tests collective communication bandwidth
- **Realistic results**: Should match InfiniBand specifications (~12.5 GB/s for 100 Gbps)

## 🔧 Environment Setup

### Conda Environment Details

The `environment.yml` file includes:
- **Python 3.12**: Latest stable Python version
- **PyTorch**: GPU-accelerated deep learning framework
- **NumPy**: Numerical computing library
- **psutil**: System and process utilities
- **mpi4py**: MPI bindings for Python (for distributed benchmarks)

### Key Dependencies

```yaml
# Core dependencies
numpy>=1.21.0
psutil>=5.8.0
pytorch>=1.12.0

# MPI for distributed computing
mpi4py>=3.1.0
openmpi>=4.1.4
```

## 📊 Understanding Results

### Performance Metrics

#### Matrix Operations
- **GFLOPS**: Billions of floating-point operations per second
- **Speedup**: How much faster GPU is compared to CPU
- **Expected ranges**:
  - **CPU**: 500-1000 GFLOPS (depending on cores)
  - **L40 GPU**: 1-2 TFLOPS
  - **H100 GPU**: 10-50 TFLOPS

#### Memory Bandwidth
- **GB/s**: Gigabytes per second
- **Expected ranges**:
  - **CPU**: 10-100 GB/s
  - **L40 GPU**: ~1.2 TB/s
  - **H100 GPU**: ~3-4 TB/s

#### Network Bandwidth
- **GB/s**: Gigabytes per second (cross-node communication)
- **Expected ranges**:
  - **100 Gbps InfiniBand**: ~12.5 GB/s
  - **200 Gbps InfiniBand**: ~25 GB/s
  - **400 Gbps InfiniBand**: ~50 GB/s
  - **Note**: These are actual network limits, not local memory bandwidth

#### Multi-GPU Scaling
- **Perfect scaling**: 2x speedup with 2 GPUs
- **Typical scaling**: 1.5-1.8x speedup (due to overhead)
- **Efficiency**: Percentage of perfect scaling achieved

### Interpreting Results

#### Good Performance Indicators
- ✅ GPU significantly faster than CPU (10-100x speedup)
- ✅ Multi-GPU shows realistic scaling (1.5-2x for 2 GPUs)
- ✅ Memory bandwidth in expected ranges
- ✅ MPI spans multiple nodes correctly

#### Warning Signs
- ❌ GPU only slightly faster than CPU (may indicate small problem size)
- ❌ Multi-GPU speedup >2x (likely false parallelism)
- ❌ Memory bandwidth >10 TB/s (calculation error)
- ❌ MPI only running on one node

## 🐛 Troubleshooting

### Common Issues

#### MPI Not Spanning Multiple Nodes
**Symptoms**: `MPI World Size: 1` on each node
**Solutions**:
1. Check SLURM configuration: `--nodes=2 --ntasks-per-node=1`
2. Verify MPI installation: `conda install -c conda-forge mpi4py openmpi`
3. Test with simple MPI script: `sbatch mpi_test.slurm`

#### GPU Memory Errors
**Symptoms**: `RuntimeError: CUDA out of memory`
**Solutions**:
1. Use `gpu_benchmark_l40.py` for memory-constrained GPUs
2. Reduce matrix sizes in scripts
3. Check GPU memory usage: `nvidia-smi`

#### CuPy Compilation Errors
**Symptoms**: `CompileException: cannot open source file`
**Solutions**:
1. Use PyTorch-based scripts (already implemented)
2. Install correct CuPy version: `pip install cupy-cuda12x`
3. Check CUDA version compatibility

### Environment Issues

#### Conda Environment Not Found
```bash
# List environments
conda env list

# Recreate environment
conda env remove -n gpu
conda env create -f environment.yml
```

#### MPI Not Available
```bash
# Install MPI
conda install -c conda-forge mpi4py openmpi

# Test installation
python -c "from mpi4py import MPI; print('MPI works!')"
```

## 🎯 Use Cases

### Single Node Testing
- **Development**: Test GPU performance during development
- **Baseline**: Establish performance baselines
- **Debugging**: Identify GPU-related issues

### Multi-GPU Testing
- **Scaling**: Understand multi-GPU performance scaling
- **Optimization**: Optimize for multi-GPU workloads
- **Capacity planning**: Determine optimal GPU configuration

### Multi-Node Testing
- **Cluster validation**: Verify cluster performance
- **Distributed training**: Test distributed machine learning setups
- **HPC applications**: Validate high-performance computing workflows

## 📈 Performance Expectations

### H100 vs L40 Comparison
- **Matrix operations**: H100 should be 5-10x faster than L40
- **Memory bandwidth**: H100 should be 2-3x faster than L40
- **Multi-GPU scaling**: Both should show similar scaling efficiency

### Real-World Performance
- **Small matrices** (<2048×2048): CPU may be competitive
- **Large matrices** (>8192×8192): GPU should dominate
- **Memory-bound operations**: GPU memory bandwidth advantage
- **Compute-bound operations**: GPU compute advantage

## 🤝 Contributing

To add new benchmarks or improve existing ones:

1. **Follow naming conventions**: `gpu_benchmark_*.py` for scripts
2. **Include SLURM script**: Create corresponding `.slurm` file
3. **Update environment**: Add new dependencies to `environment.yml`
4. **Document changes**: Update this README with new features

## 📝 License

This benchmarking suite is provided as-is for educational and research purposes. Please ensure compliance with your institution's computing policies when running benchmarks.
