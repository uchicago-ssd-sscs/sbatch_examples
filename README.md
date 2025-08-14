# Slurm Job Submission Examples

This directory contains examples of job submission scripts converted from PBS/Torque to Slurm format for use with your new Slurm cluster, including a comprehensive MPI testing suite.

## Open OnDemand

Open OnDemand is a web-based interface that provides easy access to high-performance computing resources through your web browser. It eliminates the need for command-line interaction and provides a user-friendly graphical interface for job submission, file management, and interactive computing.

### Key Features of Open OnDemand:
- **Web-based Interface**: Access HPC resources from any web browser
- **Job Submission Tool**: Submit and monitor jobs through a graphical interface
- **File Manager**: Browse, upload, download, and edit files directly in the browser
- **Interactive Apps**: Launch interactive applications.  These are in development and will soon include Jupyter notebooks, RStudio, and MATLAB
- **Job Composer**: Create and manage job scripts with templates
- **Real-time Monitoring**: View job status, output, and resource usage in real-time

### Accessing Examples in Open OnDemand

The comprehensive testing scripts and job submission examples in this repository are available in two ways:

#### 1. Terminal Access
All examples are installed in `/share/sbatch_examples` and can be accessed directly from the terminal:
```bash
# Navigate to the examples directory
cd /share/sbatch_examples

# List available examples
ls -la

# Copy examples to your home directory
cp comprehensive_test_with_mpi.sh ~/
cp comprehensive_test_no_mpi.sh ~/

# Submit jobs from terminal
sbatch comprehensive_test_with_mpi.sh
```

> The terminal can be available by connecting SSH to a logon node, or by launching the Shell Access tool in the pull-down menu of the Open OnDemand web interface.
{ .is-info }

#### 2. Open OnDemand Job Submission Tool
The examples are also available as templates in the Open OnDemand Job Submission interface:

1. **Access Job Submission**: Log into Open OnDemand and click on "Jobs" → "Job Composer"
2. **Select Template**: Choose from available templates including:
   - `comprehensive_test_no_mpi.sh` - Serial computing validation
   - `comprehensive_test_with_mpi.sh` - MPI parallel computing validation
   - Individual language examples (Fortran, Python, R, MATLAB, etc.)
3. **Customize Parameters**: Modify SLURM directives, resource requirements, and script parameters
4. **Submit Job**: Click "Submit" to launch your job through the web interface
5. **Monitor Progress**: Track job status, view output, and manage results through the dashboard

### Benefits of Using Open OnDemand:
- **No Command Line Required**: Submit jobs through an intuitive web interface
- **Template System**: Use pre-configured job templates for common tasks
- **Visual Job Management**: Monitor jobs with graphical status indicators
- **Integrated File Management**: Handle input/output files without separate file transfers
- **Cross-Platform Access**: Work from any device with a web browser
- **Educational Resources**: Built-in help and documentation for new users

### Getting Started with Open OnDemand:
1. Open your web browser and navigate to your cluster's Open OnDemand URL
2. Log in with your cluster credentials
3. Explore the available applications and tools
4. Use the Job Composer to submit your first job using the provided templates
5. Monitor job progress and view results through the web interface

## Comprehensive Testing Suites

### `comprehensive_test_no_mpi.sh`
A comprehensive testing script that validates serial computing capabilities across multiple programming languages without MPI dependencies.

**Features:**
- **Serial Testing**: Fortran, MATLAB, Python, R, and Stata
- **Built-in Functions Only**: Uses only standard libraries and built-in features
- **Comprehensive Validation**: Tests arithmetic, file I/O, data structures, and mathematical operations
- **Automatic Detection**: Detects available software and skips unavailable components
- **Detailed Logging**: Comprehensive test results and error reporting

**SLURM Configuration:**
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
```

**Usage:**
```bash
sbatch comprehensive_test_no_mpi.sh
```

**Ideal For:**
- Validating basic software installations
- Testing serial computational capabilities
- Environments without MPI support
- Quick system validation

### `comprehensive_test_with_mpi.sh`
A comprehensive testing script that validates both serial and parallel computing capabilities across multiple programming languages with extensive MPI support.

**Features:**
- **Serial Testing**: Fortran, MATLAB, Python, R, and Stata
- **MPI Parallel Testing**: MPI Fortran, MPI C, MPI Python, MPI R, MPI MATLAB
- **Performance Benchmarking**: Scalability tests, bandwidth measurements, collective operations
- **Automatic Detection**: Detects available MPI implementations and libraries
- **Comprehensive Logging**: Detailed test results and error reporting

**MPI Testing Capabilities:**
- Point-to-point communication (Send/Receive)
- Collective communication (Broadcast, Reduce, Gather, Scatter, Allreduce)
- Non-blocking communication (Isend/Irecv)
- Array operations and distributed computing
- Performance and scalability benchmarking
- Error handling and validation

**SLURM Configuration:**
```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
```

**Usage:**
```bash
sbatch comprehensive_test_with_mpi.sh
```

**Dependencies:**
- MPI implementations: OpenMPI, MPICH, or Intel MPI
- Language-specific MPI libraries: mpi4py (Python), Rmpi (R)
- Compilers: mpicc, mpif90
- Runtime: mpirun

## Key Differences Between PBS/Torque and Slurm

### Job Submission Commands
- **PBS/Torque**: `qsub script.sh`
- **Slurm**: `sbatch script.sh`

### Directive Syntax
- **PBS/Torque**: `#PBS -option value`
- **Slurm**: `#SBATCH --option=value` or `#SBATCH -option value`

### Common Directive Mappings

| PBS/Torque | Slurm | Description |
|------------|-------|-------------|
| `#PBS -N jobname` | `#SBATCH --job-name=jobname` | Job name |
| `#PBS -j oe` | `#SBATCH --output=%j.out`<br>`#SBATCH --error=%j.err` | Output/error file redirection |
| `#PBS -V` | `#SBATCH --export=ALL` | Export all environment variables |
| `#PBS -l procs=N` | `#SBATCH --ntasks=N` | Number of tasks/processes |
| `#PBS -l nodes=N:ppn=P` | `#SBATCH --nodes=N`<br>`#SBATCH --ntasks-per-node=P` | Nodes and processes per node |
| `#PBS -l mem=Xgb` | `#SBATCH --mem=XG` | Total memory for job |
| `#PBS -l pmem=Xgb` | `#SBATCH --mem-per-cpu=XG` | Memory per CPU |
| `#PBS -t 1-10` | `#SBATCH --array=1-10` | Array job specification |

### Environment Variables

| PBS/Torque | Slurm | Description |
|------------|-------|-------------|
| `$PBS_O_WORKDIR` | `$SLURM_SUBMIT_DIR` | Directory where job was submitted |
| `$PBS_JOBID` | `$SLURM_JOB_ID` | Job ID |
| `$PBS_ARRAYID` | `$SLURM_ARRAY_TASK_ID` | Array job task ID |
| `$PBS_NODEFILE` | `$SLURM_JOB_NODELIST` | File containing list of allocated nodes |
| `$PBS_NUM_PPN` | `$SLURM_NTASKS_PER_NODE` | Number of processes per node |

### MPI Commands
- **PBS/Torque**: `mpirun -n N -machinefile $PBS_NODEFILE program`
- **Slurm**: `mpirun -n N -machinefile $SLURM_JOB_NODELIST program`

### Interactive Commands

| PBS/Torque | Slurm | Description |
|------------|-------|-------------|
| `qstat` | `squeue` | View job queue and status |
| `qstat -u username` | `squeue -u username` | View jobs for specific user |
| `qstat -f jobid` | `scontrol show job jobid` | Show detailed job information |
| `qdel jobid` | `scancel jobid` | Cancel/delete a job |
| `qdel -u username` | `scancel -u username` | Cancel all jobs for a user |
| `qhold jobid` | `scontrol hold jobid` | Hold a job |
| `qrls jobid` | `scontrol release jobid` | Release a held job |
| `qrerun jobid` | `scontrol requeue jobid` | Requeue a job |
| `qsub -I` | `srun --pty bash` | Start interactive session |
| `qsub -I -l nodes=1:ppn=4` | `srun --pty --nodes=1 --ntasks-per-node=4 bash` | Interactive session with specific resources |
| `pbsnodes` | `sinfo` | View cluster/node status |
| `pbsnodes -a` | `sinfo -N` | List all nodes |
| `qstat -q` | `sinfo -p partition` | View partition/queue information |
| `qstat -B` | `sinfo -s` | View server status summary |

## Directory Structure

```
sbatch_examples/
├── comprehensive_test_no_mpi.sh      # Serial computing validation suite
├── comprehensive_test_with_mpi.sh    # MPI parallel computing validation suite
├── fortran/                          # Fortran MPI examples
├── python/                           # Python MPI and GPU examples
│   ├── openmpi/                      # OpenMPI-specific network and GPU tests
│   └── cray_mpich/                   # Cray MPICH testing scripts
├── R/                                # R MPI examples
├── matlab/                           # MATLAB parallel examples
├── stata/                            # Stata array job examples
└── container/                        # Container-based examples
    ├── apptainer/                    # Apptainer/Singularity examples
    └── podman/                       # Podman examples
```

## Available Examples

### Basic Examples
- **Fortran**: MPI hello world example (`fortran/`)
- **Python**: MPI parallel Python example (`python/`)
- **R**: R snow cluster example (`R/`)
- **MATLAB**: Parallel MATLAB example (`matlab/`)
- **Mathematica**: Parallel Mathematica example
- **Stata**: Array job example (`stata/`)

### GPU Examples
- **Multi-GPU Benchmarking**: GPU performance testing (`python/gpu_benchmark_multi_gpu.py`)
- **OpenMPI GPU Testing**: Multi-node GPU benchmarking with MPI (`python/openmpi/gpu_benchmark_mpi_multi_node.py`)

### Advanced Examples
- **Spark**: Single-node and multi-node Spark cluster examples
- **Comprehensive Serial Testing**: `comprehensive_test_no_mpi.sh` - Serial computing validation suite
- **Comprehensive MPI Testing**: `comprehensive_test_with_mpi.sh` - Full MPI validation suite
- **GPU Benchmarking**: Multi-GPU performance testing examples (`python/gpu_benchmark_multi_gpu.py`)

### MPI Testing Examples
- **MPI Fortran**: Point-to-point and collective communication (`fortran/`)
- **MPI Python**: mpi4py-based parallel computing (`python/`)
- **MPI R**: Rmpi-based statistical parallel computing (`R/`)
- **MPI MATLAB**: Parallel Computing Toolbox integration (`matlab/`)
- **Performance Testing**: Scalability and bandwidth benchmarks
- **OpenMPI Network Testing**: Specialized 100G Ethernet and network diagnostics (`python/openmpi/`)
- **Cray MPICH Testing**: Comprehensive Cray MPICH functionality and performance tests (`python/cray_mpich/`)

> **Note**: OpenMPI-specific network bandwidth and interface configuration tests are located in the `python/openmpi/` subdirectory. These include specialized scripts for 100G Ethernet testing and network diagnostics.

## Usage

1. Copy the example scripts to your home directory
2. Modify the scripts as needed for your specific application
3. Submit jobs using `sbatch script.sh`

> **Important**: Specialized MPI testing scripts are available in dedicated subdirectories:
> - **OpenMPI Network Testing** (`python/openmpi/`):
>   - `mpi_test_with_100G_ethernet.slurm` - Specialized 100G Ethernet MPI testing
>   - `slurm_diagnostic.sh` - SLURM and MPI configuration diagnostics
>   - `mpi_test.py` - Basic MPI functionality test
>   - `network_bandwidth_test_v2.py` - Advanced network bandwidth testing
>   - `gpu_benchmark_mpi_multi_node.py` - Multi-node GPU benchmarking with MPI
> - **Cray MPICH Testing** (`python/cray_mpich/`):
>   - `cray_mpich_test.py` - Comprehensive Cray MPICH functionality and performance tests
>   - `cray_mpich_test.slurm` - SLURM batch script for Cray MPICH testing

### For Comprehensive Testing:
```bash
# Submit serial computing test (no MPI required)
sbatch comprehensive_test_no_mpi.sh

# Submit comprehensive MPI test
sbatch comprehensive_test_with_mpi.sh

# Check test results
cat test_output/comprehensive_test.log
cat test_output/test_results.txt
```

## Notes

- All examples include proper shebang lines (`#!/bin/bash`)
- Memory specifications use capital G (e.g., `50G` instead of `50gb`)
- Array jobs use `SLURM_ARRAY_TASK_ID` instead of `PBS_ARRAYID`
- The `--export=ALL` directive ensures all environment variables are available to the job
- Output and error files use `%j` placeholder which gets replaced with the job ID
- Both comprehensive test scripts automatically detect available software and skip unavailable components
- Comprehensive logging provides detailed test results and performance metrics
- Choose `comprehensive_test_no_mpi.sh` for basic validation or `comprehensive_test_with_mpi.sh` for full parallel testing

## Additional Useful Slurm Commands

### Job Management
- `squeue --start` - Show estimated start times for pending jobs
- `squeue --format="%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"` - Custom job queue format
- `squeue -t RUNNING` - Show only running jobs
- `squeue -t PENDING` - Show only pending jobs
- `squeue -t COMPLETED` - Show recently completed jobs
- `sacct -j jobid` - Show accounting data for a job
- `sacct -u username --starttime=YYYY-MM-DD` - Show user's job history

### Resource Information
- `sinfo -l` - Long format cluster status
- `sinfo -r` - Show only running nodes
- `sinfo -t idle` - Show only idle nodes
- `sinfo -t down` - Show only down nodes
- `scontrol show partition partition_name` - Show partition details
- `scontrol show node node_name` - Show specific node details

### Interactive Sessions
- `srun --pty --gres=gpu:1 bash` - Interactive session with GPU
- `srun --pty --mem=16G bash` - Interactive session with specific memory
- `srun --pty --time=02:00:00 bash` - Interactive session with time limit

### Job Control
- `scontrol update job jobid TimeLimit=01:00:00` - Modify job time limit
- `scontrol update job jobid Partition=partition_name` - Move job to different partition
- `scontrol requeue jobid` - Requeue a job (useful after system issues)
- `scontrol hold jobid` - Hold a job (prevent it from starting)
- `scontrol release jobid` - Release a held job

## Comprehensive Testing Output

Both comprehensive testing scripts generate:
- **Test Results**: Detailed output from each test in `test_output/test_results.txt`
- **Error Logs**: Error details in `test_output/error_details.log`
- **Summary Log**: Overall test summary in `test_output/comprehensive_test.log`
- **Performance Data**: Timing and scalability metrics (MPI script includes additional parallel performance data) 