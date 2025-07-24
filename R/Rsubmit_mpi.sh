####################################################
#Copy this file to your home directory and run it with sbatch
#"sbatch Rsubmit.sh" will start the job 
#This script will start 25 instances of R in a snow cluster
#You will need to add "module load openmpi/1.10.7" to your .profile first
###################################################

#!/bin/bash
#SBATCH --job-name=Rsnow
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL
#SBATCH --ntasks=25
#SBATCH --mem=50G

cd $SLURM_SUBMIT_DIR

mpirun -n 1 -machinefile $SLURM_JOB_NODELIST R CMD BATCH Rprogram.R Rprogram.Rout
