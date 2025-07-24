##################################################
#Copy this file to your home directory and run it with sbatch
#"sbatch matlabparallelbatch.sh" will start the job
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

cd $SLURM_SUBMIT_DIR

#Matlab can clobber it's temporary files if multiple instances are run at once
#Create a job specific temp directory to avoid this
mkdir -p ~/matlabtmp/$SLURM_JOB_ID  
export MATLABWORKDIR=~/matlabtmp/$SLURM_JOB_ID

matlab -nodesktop -nosplash  -r matlab_input >> stdout.log

#Delete the temporary directory
#rm -rf $MATLABWORKDIR
sleep 10000
