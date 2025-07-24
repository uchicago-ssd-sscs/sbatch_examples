##################################################
#Copy this file to your home directory and run it with sbatch
#"sbatch stata-array.sh" will start the job
#stata-mp is licensed for 8 cores
#This script will start 10 stata-mp jobs each running on 8 cores
###################################################

#!/bin/bash
#SBATCH --job-name=StatmpArray
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=10G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL
#SBATCH --array=1-10

cd $SLURM_SUBMIT_DIR
# execute program
# this will start 10 stata jobs 
# loading files HelloHello-1.do, HelloWorld-2.do, HelloWorld-3.do, ...
stata-mp -b do HelloWorld-${SLURM_ARRAY_TASK_ID}.do
