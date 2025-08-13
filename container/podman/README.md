# BOINC Distributed Computing with Podman

This example demonstrates how to run BOINC (Berkeley Open Infrastructure for Network Computing) distributed computing tasks using Podman containers in a SLURM cluster environment.

## What is BOINC?

BOINC is a platform for volunteer computing that allows you to contribute your computer's processing power to scientific research projects. This setup uses the Rosetta@home project, which focuses on protein structure prediction and design.

## Features

- **Containerized BOINC**: Uses Podman to run BOINC in an isolated container
- **SLURM Integration**: Properly integrates with SLURM job scheduling
- **Resource Management**: Configurable CPU and memory limits
- **Data Persistence**: BOINC data is preserved between job runs
- **Monitoring**: Regular status reports and progress tracking
- **Automatic Cleanup**: Optional cleanup of data after job completion

## Prerequisites

1. **SLURM Cluster**: Access to a SLURM cluster with CPU partition
2. **Podman**: Podman must be available on the cluster nodes
3. **Internet Access**: Cluster nodes need access to BOINC project servers
4. **Storage**: Sufficient disk space for BOINC work units and data

## Setup

### 1. Verify Podman Availability

```bash
# Check if Podman is available
podman --version

# If not available, contact your system administrator
```

### 2. Submit the Job

```bash
sbatch boinc.slurm
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

Modify the SLURM parameters in `boinc.slurm`:
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
tail -f <job_id>_boinc.out

# Check job logs
tail -f <job_id>_boinc.err
```

### Check Results

After the job completes, you'll find:

- **BOINC data directory**: `boinc_data_<job_id>/` containing all BOINC files
- **Summary file**: `boinc_summary_<job_id>.txt` with job statistics
- **Log files**: SLURM output and error files

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

1. **Podman Not Available**
   ```
   ERROR: Podman is not available on this system!
   ```
   Solution: Contact your system administrator to install Podman.

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

### Debug Mode

To run in debug mode, modify the verbosity in `boinc_config.xml`:

```xml
<verbosity>99</verbosity>
```

## Performance Considerations

- **CPU Usage**: BOINC will use all allocated CPUs
- **Memory Usage**: Typically 1-2GB per CPU core
- **Disk Usage**: 1-10GB depending on project and work buffer
- **Network Usage**: Periodic downloads/uploads of work units
- **Duration**: Work units can take hours to days to complete

## Security Notes

- BOINC runs in an isolated container
- No root access required
- Data is contained within the job directory
- Network access is limited to BOINC project servers

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

### Cleanup

To automatically clean up data after job completion, uncomment the cleanup line in the script:

```bash
# echo "Cleaning up BOINC data directory..."
# rm -rf $BOINC_DATA_DIR
```

## Contributing to Science

By running this BOINC job, you're contributing to scientific research:

- **Rosetta@home**: Helps design new proteins for medical applications
- **SETI@home**: Searches for signs of intelligent life in the universe
- **Climate Prediction**: Improves climate models and predictions
- **World Community Grid**: Supports humanitarian research projects

## License

This project is part of the sbatch_examples repository of the Social Sciences Computing Services team at the University of Chicago.
