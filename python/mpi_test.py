#!/usr/bin/env python3

import platform
import os

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    exit(1)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get node information
    hostname = platform.node()
    
    # Print basic MPI info
    print(f"Rank {rank}/{size} on {hostname}")
    print(f"Process ID: {os.getpid()}")
    
    # Synchronize all processes
    comm.Barrier()
    
    # Gather all hostnames to rank 0
    all_hostnames = comm.gather(hostname, root=0)
    
    if rank == 0:
        print(f"\nMPI Test Results:")
        print(f"Total processes: {size}")
        print(f"Hostnames: {all_hostnames}")
        print(f"Unique nodes: {len(set(all_hostnames))}")
        
        if len(set(all_hostnames)) > 1:
            print("✅ SUCCESS: MPI is spanning multiple nodes!")
        else:
            print("❌ FAILURE: MPI is only running on one node")
    
    # Test simple communication
    if size > 1:
        if rank == 0:
            # Send message to rank 1
            message = f"Hello from rank 0 on {hostname}"
            comm.send(message, dest=1, tag=123)
            print(f"Rank 0: Sent message to rank 1")
        elif rank == 1:
            # Receive message from rank 0
            message = comm.recv(source=0, tag=123)
            print(f"Rank 1: Received message: {message}")
    
    comm.Barrier()
    
    if rank == 0:
        print("\nMPI test completed!")

if __name__ == "__main__":
    main()
