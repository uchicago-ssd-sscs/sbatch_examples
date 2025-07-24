This is a fortran MPI hello world script. 

1) Copy the example scripts to your home directory

cp -rp /share/sbatch_examples/fortran ~/
cd ~/fortran

2) Add the following line to a file called .profile in your home directory.
This will enable OpenMPI for your account. 

module load openmpi/1.10.7

3) Compile the example program

mpif90 -o mpi_hello mpi_hello.f

4) Submit the job to the cluster with sbatch. 

sbatch fortransubmit.sh
