# BOINC Distributed Computing with Apptainer

This example demonstrates how to run BOINC (Berkeley Open Infrastructure for Network Computing) distributed computing tasks using Apptainer containers in a SLURM cluster environment.

## What is BOINC?

BOINC is a platform for volunteer computing that allows you to contribute your computer's processing power to scientific research projects. This setup uses the Rosetta@home project, which focuses on protein structure prediction and design.

## What is Apptainer?

Apptainer (formerly Singularity) is a container runtime designed specifically for high-performance computing (HPC) environments. It provides secure, reproducible environments that can run on shared computing resources without requiring root privileges.

## Features

- **Containerized BOINC**: Uses Apptainer to run BOINC in an isolated container
- **SLURM Integration**: Properly integrates with SLURM job scheduling
- **Resource Management**: Configurable CPU and memory limits
- **Data Persistence**: BOINC data is preserved between job runs
- **Monitoring**: Regular status reports and progress tracking
- **Automatic Cleanup**: Optional cleanup of data after job completion
- **HPC Optimized**: Designed for shared computing environments

## Prerequisites

1. **SLURM Cluster**: Access to a SLURM cluster with CPU partition
2. **Apptainer**: Apptainer must be available on the cluster nodes
3. **Internet Access**: Cluster nodes need access to BOINC project servers
4. **Storage**: Sufficient disk space for BOINC work units and data

## Setup

### 1. Verify Apptainer Availability

```bash
# Check if Apptainer is available
apptainer --version

# If not available, contact your system administrator
```

### 2. Submit the Job

```bash
sbatch apptainer.slurm
```

## Configuration

### Default Settings

The job is configured with:
- **1 node** in the CPU partition
- **4 CPUs** per task
- **8GB memory** per node
- **24-hour time limit**
- **Rosetta@home project** (protein structure prediction)

### Customization Options

#### Change BOINC Project

Edit the project URL in the SLURM script:

```bash
# Popular BOINC projects:
BOINC_PROJECT_URL="https://boinc.bakerlab.org/rosetta/"     # Protein folding
BOINC_PROJECT_URL="https://setiathome.berkeley.edu/"       # Search for ET
BOINC_PROJECT_URL="https://climateprediction.net/"         # Climate modeling
BOINC_PROJECT_URL="https://www.worldcommunitygrid.org/"    # Multiple projects
```

#### Adjust Resources

Modify the SLURM parameters in `apptainer.slurm`:
- `--cpus-per-task`: Number of CPUs to allocate
- `--mem`: Memory allocation per node
- `--time`: Job time limit
- `--nodes`: Number of nodes (for multi-node setup)

#### BOINC Configuration

The script creates a `boinc_config.xml` file with settings like:
- CPU usage limits
- Memory limits
- Disk space limits
- Network settings
- Work buffer settings

## Usage

### Monitor the Job

```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f <job_id>_boinc_apptainer.out

# Check job logs
tail -f <job_id>_boinc_apptainer.err
```

### Check Results

After the job completes, you'll find:

- **BOINC data directory**: `boinc_data_<job_id>/` containing all BOINC files
- **Summary file**: `boinc_apptainer_summary_<job_id>.txt` with job statistics
- **Log files**: SLURM output and error files
- **Container image**: `boinc_client.sif` (reusable for future jobs)

### Access BOINC Data

```bash
# Navigate to BOINC data directory
cd boinc_data_<job_id>/

# View BOINC logs
cat boinc.log

# Check work unit status
ls -la slots/

# View project files
ls -la projects/
```

## Apptainer vs Podman

### Key Differences

| Feature | Apptainer | Podman |
|---------|-----------|--------|
| **HPC Focus** | Designed for HPC environments | General-purpose container runtime |
| **Security Model** | User namespace support | Rootless containers |
| **Image Format** | SIF (Singularity Image Format) | OCI/Docker format |
| **Privilege Requirements** | No root required | No root required |
| **Network Isolation** | Limited | Full network isolation |
| **Performance** | Optimized for HPC workloads | General performance |

### Advantages of Apptainer

1. **HPC Optimized**: Designed specifically for shared computing environments
2. **No Root Required**: Can run on systems where users don't have root access
3. **SIF Format**: Efficient, immutable container images
4. **GPU Support**: Excellent GPU passthrough capabilities
5. **MPI Support**: Native support for MPI applications

## BOINC Projects

### Rosetta@home (Default)
- **Purpose**: Protein structure prediction and design
- **Impact**: Drug discovery, disease research
- **Requirements**: Moderate CPU usage

### Other Popular Projects

1. **SETI@home**: Search for extraterrestrial intelligence
2. **Climate Prediction**: Climate modeling and prediction
3. **World Community Grid**: Multiple humanitarian projects
4. **Folding@home**: Protein folding for disease research
5. **PrimeGrid**: Mathematical research

## Troubleshooting

### Common Issues

1. **Apptainer Not Available**
   ```
   ERROR: Apptainer is not available on this system!
   ```
   Solution: Contact your system administrator to install Apptainer.

2. **Container Pull Failure**
   ```
   Error pulling BOINC container image
   ```
   Solution: Check internet connectivity and Docker registry access.

3. **BOINC Project Connection Issues**
   ```
   Failed to attach to project
   ```
   Solution: Verify network connectivity to BOINC project servers.

4. **Insufficient Disk Space**
   ```
   Disk space limit exceeded
   ```
   Solution: Increase disk space allocation or reduce work buffer.

5. **Memory Limit Exceeded**
   ```
   Container killed due to memory limit
   ```
   Solution: Increase memory allocation in SLURM script.

6. **Permission Denied**
   ```
   Permission denied when accessing container
   ```
   Solution: Ensure proper file permissions and Apptainer configuration.

### Debug Mode

To run in debug mode, modify the verbosity in `boinc_config.xml`:

```xml
<verbosity>99</verbosity>
```

### Apptainer Debugging

```bash
# Check Apptainer configuration
apptainer config list

# Test container execution
apptainer exec boinc_client.sif /bin/bash

# Check container info
apptainer inspect boinc_client.sif
```

## Performance Considerations

- **CPU Usage**: BOINC will use all allocated CPUs
- **Memory Usage**: Typically 1-2GB per CPU core
- **Disk Usage**: 1-10GB depending on project and work buffer
- **Network Usage**: Periodic downloads/uploads of work units
- **Duration**: Work units can take hours to days to complete
- **Container Overhead**: Minimal with Apptainer

## Security Notes

- BOINC runs in an isolated Apptainer container
- No root access required
- Data is contained within the job directory
- Network access is limited to BOINC project servers
- Apptainer provides additional security isolation

## Multi-Node Setup

To run BOINC across multiple nodes, modify the SLURM script:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
```

Each node will run its own BOINC container instance.

## Data Management

### Preserving Data

BOINC data is automatically preserved in the job directory. To reuse data:

```bash
# Copy data from previous job
cp -r boinc_data_<previous_job_id> boinc_data_<new_job_id>
```

### Container Image Management

The container image (`boinc_client.sif`) is cached locally and reused:

```bash
# Check image size
ls -lh boinc_client.sif

# Remove old image to force re-download
rm boinc_client.sif
```

### Cleanup

To automatically clean up data after job completion, uncomment the cleanup line in the script:

```bash
# echo "Cleaning up BOINC data directory..."
# rm -rf $BOINC_DATA_DIR
```

## Advanced Apptainer Features

### Custom Container Definition

Create a custom Apptainer definition file:

```bash
# Create definition file
cat > boinc.def << EOF
Bootstrap: docker
From: boinc/client:latest

%environment
    export BOINC_PROJECT_URL="https://boinc.bakerlab.org/rosetta/"
    export BOINC_USERNAME="anonymous"
    export BOINC_PASSWORD="anonymous"

%runscript
    exec boinc "\$@"
EOF

# Build custom container
apptainer build boinc_custom.sif boinc.def
```

### GPU Support

To enable GPU support, add GPU flags to the Apptainer command:

```bash
apptainer exec --nv $CONTAINER_IMAGE /var/lib/boinc/run_boinc.sh
```

### MPI Support

For MPI applications, use the `--bind` flag to mount MPI libraries:

```bash
apptainer exec --bind /usr/lib64/mpi:/usr/lib64/mpi $CONTAINER_IMAGE command
```

## Contributing to Science

By running this BOINC job, you're contributing to scientific research:

- **Rosetta@home**: Helps design new proteins for medical applications
- **SETI@home**: Searches for signs of intelligent life in the universe
- **Climate Prediction**: Improves climate models and predictions
- **World Community Grid**: Supports humanitarian research projects

## License

This project is part of the sbatch_examples repository of the Social Sciences Computing Services team at the University of Chicago.
