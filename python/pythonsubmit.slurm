#!/bin/bash
#SBATCH --job-name=PythonTest
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --export=ALL
#SBATCH --ntasks=50
#SBATCH --mem=50G

#Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

#Load Python Anaconda 
#module load python/anaconda3

# Activate conda environment from existing installation in home directory (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate gpu || echo "Warning: could not activate conda env 'gpu'"
else
    echo "Warning: conda.sh not found; skipping conda activation"
fi

#You may load a virtual environment 
#source activate myenvironment

#srun will start 49 instances of helloworld.py
#helloworld.py will print the jobs rank
srun -n 49 python helloworld.py 