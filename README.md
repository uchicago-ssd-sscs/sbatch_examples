# Slurm Job Submission Examples

This directory contains examples of job submission scripts converted from PBS/Torque to Slurm format for use with your new Slurm cluster.

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

## Available Examples

### Basic Examples
- **Fortran**: MPI hello world example
- **Python**: MPI parallel Python example
- **R**: R snow cluster example
- **MATLAB**: Parallel MATLAB example
- **Mathematica**: Parallel Mathematica example
- **Stata**: Array job example

### Advanced Examples
- **Spark**: Single-node and multi-node Spark cluster examples

## Usage

1. Copy the example scripts to your home directory
2. Modify the scripts as needed for your specific application
3. Submit jobs using `sbatch script.sh`

## Notes

- All examples include proper shebang lines (`#!/bin/bash`)
- Memory specifications use capital G (e.g., `50G` instead of `50gb`)
- Array jobs use `SLURM_ARRAY_TASK_ID` instead of `PBS_ARRAYID`
- The `--export=ALL` directive ensures all environment variables are available to the job
- Output and error files use `%j` placeholder which gets replaced with the job ID

## Additional Slurm Commands

- `squeue` - View job queue (equivalent to `qstat`)
- `scancel jobid` - Cancel a job (equivalent to `qdel`)
- `sinfo` - View cluster status
- `scontrol show job jobid` - Show detailed job information 