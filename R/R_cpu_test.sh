#!/bin/bash
#SBATCH --job-name=R_CPU_Test
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load R module (adjust module name as needed for your system)
module load R

# Run the R script
R CMD BATCH R_cpu_test.R R_cpu_test.Rout

echo "R CPU test completed. Check R_cpu_test.Rout for results." 