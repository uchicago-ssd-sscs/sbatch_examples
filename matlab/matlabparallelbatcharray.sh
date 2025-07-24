##################################################
#Copy this file to your home directory and run it with sbatch
#"sbatch matlab.sh" will start the job
#This script is intended to reserve 12 processors for Matlab worker processes
#In your batch script you can run "parpool('local',12)" to start 12 workers on a node
###################################################

#!/bin/bash
#SBATCH --job-name=matlabpool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --array=1-10

cd $SLURM_SUBMIT_DIR

#Matlab can clobber it's temporary files if multiple instances are run at once
#Create a job specific temp directory to avoid this
mkdir -p ~/matlabtmp/$SLURM_JOB_ID
export MATLABWORKDIR=~/matlabtmp/$SLURM_JOB_ID


# execute program
# this will start 10 matlab jobs each requesting 12 cores
# loading files matlab_input1.m, matlab_input2.m, matlab_input3.m, ... matlab_input10.m
matlab -nodesktop -nosplash  -r matlab_input${SLURM_ARRAY_TASK_ID} >> stdout.log

#Delete the temporary directory
#rm -rf $MATLABWORKDIR
