# OpenMPI Network Testing and Multi-Node GPU Benchmarking

This directory contains specialized scripts for testing OpenMPI network performance, multi-node GPU coordination, and cluster diagnostics. These tools are designed for **Cluster Administrators**, **IT Support Staff**, and **HPC Researchers** who need to validate network performance and distributed computing capabilities.

> OpenMPI performance is not optimal on this cluster.  Use the HPE/Cray MPI examples in the "cray_mpich" directory for better performance.
{ .is-warning }

## 🎯 **Target Audience**

### **Cluster Administrators**
- **Purpose**: Validate network infrastructure and multi-node performance
- **Focus**: Network bandwidth testing, MPI configuration, cluster diagnostics
- **Benefits**: Identify network bottlenecks, optimize MPI settings, ensure cluster reliability

### **IT Support Staff**
- **Purpose**: Troubleshoot network and MPI issues
- **Focus**: Diagnostic tools, performance monitoring, configuration validation
- **Benefits**: Quick problem identification, performance baseline establishment

### **HPC Researchers**
- **Purpose**: Test distributed computing capabilities
- **Focus**: Multi-node GPU coordination, network-aware applications
- **Benefits**: Validate distributed workflows, optimize multi-node performance

## 🚀 **Quick Start Guide**

### **Step 1: Install Miniconda**

First, install Miniconda in your home directory:

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda (accept defaults for home directory installation)
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell or source bashrc
source ~/.bashrc
```

### **Step 2: Create OpenMPI Environment**

Create the OpenMPI environment from the provided `environment.yml` in this directory:

```bash
# Create environment from YAML file
conda env create -f environment.yml -n openmpi

# Activate the environment
conda activate openmpi
```

### **Step 3: Run Network Tests**

Choose the appropriate test for your needs:

```bash
# Basic MPI functionality test
sbatch mpi_test.slurm

# Network bandwidth testing
sbatch mpi_test_with_100G_ethernet.slurm

# Multi-node GPU benchmarking
sbatch gpu_benchmark_multi.slurm

# System diagnostics
./slurm_diagnostic.sh
```

## 📊 **Detailed Script Explanations**

### **Basic MPI Testing**

#### `mpi_test.py` & `mpi_test.slurm`
**What it does**: Verifies that MPI is working correctly across multiple nodes

**Real-world analogy**:
- Imagine testing if all computers in an office can talk to each other
- Each computer sends a message to every other computer
- We verify that messages are received correctly and on time

**What it tests**:
- **MPI Installation**: Ensures MPI is properly installed and configured
- **Multi-Node Communication**: Verifies nodes can communicate with each other
- **Network Connectivity**: Tests basic network connectivity between nodes
- **MPI Configuration**: Validates MPI settings and environment variables

**Expected results**:
- All nodes can communicate with each other
- MPI spans multiple nodes (not just one node)
- Network latency is within expected ranges

**Learn more**:
- [MPI Basics](https://en.wikipedia.org/wiki/Message_Passing_Interface)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [MPI4Py Tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html)

### **Network Bandwidth Testing**

#### `network_bandwidth_test_v2.py` & `mpi_test_with_100G_ethernet.slurm`
**What it does**: Measures actual network bandwidth between nodes

**Real-world analogy**:
- Imagine testing how fast you can transfer files between computers
- We measure how much data can be sent per second
- This tells us the actual network speed, not just the advertised speed

**What it tests**:
- **Point-to-Point Bandwidth**: Data transfer between specific node pairs
- **Bidirectional Bandwidth**: Simultaneous send/receive performance
- **All-to-All Bandwidth**: Collective communication performance
- **Network Interface Detection**: Identifies which network interfaces are being used

**Expected results**:
- **100 Gbps InfiniBand**: ~12.5 GB/s bandwidth
- **200 Gbps InfiniBand**: ~25 GB/s bandwidth
- **400 Gbps InfiniBand**: ~50 GB/s bandwidth
- **1 Gbps Ethernet**: ~125 MB/s bandwidth

**Why it matters**:
- **Cluster Admins**: Verify network infrastructure performance
- **IT Support**: Identify network bottlenecks and configuration issues
- **Researchers**: Understand network limitations for distributed applications

**Learn more**:
- [InfiniBand Technology](https://en.wikipedia.org/wiki/InfiniBand)
- [Network Bandwidth Testing](https://en.wikipedia.org/wiki/Bandwidth_test)
- [MPI Network Performance](https://www.open-mpi.org/faq/?category=network)

### **Multi-Node GPU Benchmarking**

#### `gpu_benchmark_mpi_multi_node.py` & `gpu_benchmark_multi.slurm`
**What it does**: Tests GPU performance across multiple nodes using MPI coordination

**Real-world analogy**:
- Imagine coordinating work across multiple teams in different offices
- Each team has powerful tools (GPUs) but needs to work together
- We measure how well they can coordinate and share results

**What it tests**:
- **Distributed Matrix Operations**: Matrix multiplication across multiple nodes
- **Multi-Node Memory Bandwidth**: Data transfer between nodes with GPU acceleration
- **GPU Coordination**: How well GPUs on different nodes can work together
- **Network-GPU Integration**: Performance of GPU operations that require network communication

**Expected results**:
- **Single Node**: Baseline GPU performance
- **Multi-Node**: Performance scaling with additional nodes
- **Network Overhead**: Impact of network communication on GPU performance
- **Coordination Efficiency**: How well nodes can coordinate GPU operations

**Why it matters**:
- **Researchers**: Validate distributed machine learning workflows
- **Cluster Admins**: Understand multi-node GPU performance characteristics
- **IT Support**: Identify bottlenecks in distributed GPU applications

**Learn more**:
- [Distributed Deep Learning](https://en.wikipedia.org/wiki/Distributed_computing)
- [GPU Computing](https://developer.nvidia.com/gpu-computing)
- [Multi-Node Training](https://pytorch.org/docs/stable/distributed.html)

### **System Diagnostics**

#### `slurm_diagnostic.sh`
**What it does**: Comprehensive system diagnostics for SLURM and MPI configuration

**Real-world analogy**:
- Imagine a comprehensive health check for your computer network
- We check all the vital signs: network connections, software versions, configurations
- This helps identify problems before they become serious

**What it checks**:
- **SLURM Configuration**: Job scheduler settings and partition configuration
- **MPI Installation**: MPI version, configuration, and environment variables
- **Network Interfaces**: Available network interfaces and their configurations
- **GPU Configuration**: GPU detection, drivers, and CUDA installation
- **System Resources**: Memory, CPU, and storage availability

**Expected results**:
- Complete system health report
- Configuration recommendations
- Performance baseline establishment
- Problem identification and troubleshooting guidance

**Why it matters**:
- **Cluster Admins**: Proactive system monitoring and maintenance
- **IT Support**: Quick problem diagnosis and resolution
- **Researchers**: Understanding system capabilities and limitations

**Learn more**:
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [System Monitoring](https://en.wikipedia.org/wiki/System_monitoring)
- [HPC Best Practices](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/high-performance-computing/)

## 🔧 **Technical Requirements**

### **System Requirements**
- **Network**: High-speed interconnect (InfiniBand recommended)
- **Nodes**: Multiple compute nodes with network connectivity
- **GPUs**: NVIDIA GPUs with CUDA support (for GPU tests)
- **SLURM**: Job scheduler for multi-node coordination
- **Storage**: Sufficient space for temporary files and results

### **Software Dependencies**
- **OpenMPI**: MPI implementation for distributed computing
- **PyTorch**: GPU computing framework with CUDA support
- **NumPy**: Numerical computing library
- **psutil**: System monitoring utilities
- **NetworkX**: Network analysis tools

### **Performance Expectations**
- **Network Bandwidth**: 10-50 GB/s for high-speed interconnects
- **MPI Latency**: <1 microsecond for low-latency networks
- **GPU Scaling**: 1.5-2x speedup with 2 nodes, diminishing returns with more nodes
- **Coordination Overhead**: 5-20% overhead for multi-node coordination

## 🐛 **Troubleshooting Common Issues**

### **MPI Not Spanning Multiple Nodes**
**Symptoms**: `MPI World Size: 1` on each node
**Solutions**:
1. Check SLURM configuration: `--nodes=2 --ntasks-per-node=1`
2. Verify MPI installation: `conda install -c conda-forge mpi4py openmpi`
3. Test with simple MPI script: `sbatch mpi_test.slurm`

### **Low Network Bandwidth**
**Symptoms**: Bandwidth significantly below expected values
**Solutions**:
1. Check network interface configuration
2. Verify InfiniBand drivers and firmware
3. Test with different message sizes
4. Check for network congestion or configuration issues

### **GPU Not Available on Multiple Nodes**
**Symptoms**: GPU tests only run on single node
**Solutions**:
1. Verify GPU allocation in SLURM script
2. Check CUDA installation on all nodes
3. Ensure PyTorch CUDA support is available
4. Test GPU detection: `nvidia-smi`

### **MPI Configuration Issues**
**Symptoms**: MPI jobs fail to start or communicate
**Solutions**:
1. Check MPI environment variables
2. Verify network interface selection
3. Test with different MPI implementations
4. Review SLURM and MPI integration

## 📈 **Performance Optimization**

### **Network Optimization**
- **Use High-Speed Interconnects**: InfiniBand or high-speed Ethernet
- **Optimize Message Sizes**: Test different message sizes for optimal performance
- **Reduce Network Contention**: Avoid running multiple bandwidth tests simultaneously
- **Monitor Network Utilization**: Use network monitoring tools

### **MPI Configuration**
- **Choose Appropriate MPI Implementation**: OpenMPI, MPICH, or vendor-specific
- **Optimize MPI Parameters**: Buffer sizes, collective algorithms, network selection
- **Use Appropriate Network Interfaces**: Select fastest available network
- **Monitor MPI Performance**: Use MPI profiling tools

### **GPU Coordination**
- **Optimize Data Distribution**: Minimize data transfer between nodes
- **Use Efficient Communication Patterns**: Reduce communication overhead
- **Monitor GPU Utilization**: Ensure GPUs are not idle during communication
- **Balance Computation and Communication**: Overlap computation with communication

## 🎯 **Usage Examples**

### **Quick Network Validation**
```bash
# Test basic MPI functionality
sbatch mpi_test.slurm

# Check results
cat slurm-*.out
```

### **Comprehensive Network Testing**
```bash
# Run full network bandwidth test
sbatch mpi_test_with_100G_ethernet.slurm

# Monitor progress
squeue -u $USER
tail -f slurm-*.out
```

### **Multi-Node GPU Benchmarking**
```bash
# Run distributed GPU benchmark
sbatch gpu_benchmark_multi.slurm

# Check GPU utilization across nodes
srun --nodes=2 nvidia-smi
```

### **System Diagnostics**
```bash
# Run comprehensive diagnostics
./slurm_diagnostic.sh

# Review diagnostic output
cat diagnostic_report.txt
```

## 📝 **Key Takeaways**

### **Network Performance**
- **Bandwidth Testing**: Essential for validating network infrastructure
- **Latency Measurement**: Critical for distributed applications
- **Interface Selection**: Important for optimal performance
- **Configuration Validation**: Ensures proper network setup

### **Multi-Node Coordination**
- **MPI Communication**: Foundation for distributed computing
- **GPU Coordination**: Enables distributed machine learning
- **Performance Scaling**: Understanding multi-node performance characteristics
- **Overhead Management**: Balancing computation and communication

### **System Management**
- **Proactive Monitoring**: Identify issues before they impact users
- **Performance Baselines**: Establish performance expectations
- **Configuration Optimization**: Fine-tune system settings
- **Troubleshooting Tools**: Quick problem identification and resolution

## 🤝 **Support and Documentation**

### **Getting Help**
- **Performance Issues**: Check network utilization and MPI configuration
- **Configuration Problems**: Verify SLURM settings and MPI installation
- **Hardware Issues**: Test with diagnostic tools and monitoring

### **Further Resources**
- [OpenMPI Documentation](https://www.open-mpi.org/doc/) - Complete MPI programming guide
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html) - Job scheduling and cluster management
- [NVIDIA GPU Computing](https://developer.nvidia.com/gpu-computing) - GPU programming and optimization
- [HPC Best Practices](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/high-performance-computing/) - High-performance computing guidelines

### **Community Support**
- [OpenMPI Users Mailing List](https://www.open-mpi.org/community/lists/) - MPI community discussions
- [SLURM Users Mailing List](https://slurm.schedmd.com/mail.html) - SLURM community support
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) - GPU computing discussions
- [HPC Stack Exchange](https://hpc.stackexchange.com/) - HPC community Q&A

This comprehensive testing suite provides the tools needed to validate and optimize network performance and multi-node GPU coordination in HPC environments.
