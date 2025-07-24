This example will spawn 100 instances of Python across the cluster with MPI and report the process rank. 

We recomend using the Anaconda distribution of Python
You will need to load Anaconda, mpi4py is already installed 
Run these commands in a terminal to set up your environment

user@acropolis ~$ module load python/anaconda3

Copy helloworld.py  pythonsubmit.qsub to your home directory and run it with sbatch

user@acropolis ~$ sbatch pythonsubmit.qsub 
