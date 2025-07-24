#Load snow and Rmpi packages
library(snow)
library(Rmpi)
#Spawn 25 workers based off "#PBS -l procs="
cl <- makeCluster(mpi.universe.size()-1, type = "MPI")
clusterCall(cl, function() Sys.info()[c("nodename","machine")])
clusterCall(cl, runif, mpi.universe.size()-1)
#Shutdown the worker processes and quit MPI
stopCluster(cl)
mpi.quit()
