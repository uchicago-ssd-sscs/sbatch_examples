##################################################
#Copy this file to your home directory and run it with qsub
#"qsub stata-mp.sh" will start the job  
#stata-mp is licensed for 8 cores 
#This script will start 1 stata15 job running on 8 cores 
###################################################

#!/bin/bash
#PBS -N Statamp 
#Stata15 mp is licensed for 8 CPU's 
#PBS -l nodes=1:ppn=8,mem=10gb
#PBS -j oe
#
cd $PBS_O_WORKDIR
# execute program
stata-mp -b do HelloWorld.do
