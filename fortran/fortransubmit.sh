#!/bin/bash
#SBATCH --job-name=FortranTest
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL
#SBATCH --ntasks=25
#SBATCH --mem=10G

cd $SLURM_SUBMIT_DIR

#mpirun will start 25 instances of mpi_hello 
#$SLURM_JOB_NODELIST tells mpirun which CPU's Slurm reserved for the job
#mpi_hello will print the jobs rank
mpirun -n 25 -machinefile $SLURM_JOB_NODELIST mpi_hello 
