#!/usr/bin/env python3

import subprocess
import platform
import os
import sys

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("MPI not available")
    sys.exit(1)

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_network_interfaces():
    """Check network interface configuration"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== Network Interface Configuration ===")
        
        # Check all network interfaces
        print("Network interfaces:")
        print(run_command("ip addr show"))
        print()
        
        # Check bond interfaces specifically
        print("Bond interface details:")
        for bond in ['bond0', 'bond1', 'bond2']:
            print(f"\n{bond}:")
            print(run_command(f"ethtool {bond}"))
        print()
        
        # Check routing table
        print("Routing table:")
        print(run_command("ip route show"))
        print()

def check_mpi_network_config():
    """Check MPI network configuration"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== MPI Network Configuration ===")
        
        # Check MPI environment variables
        mpi_vars = ['MPICH_NETMASK', 'MPICH_DEVICE', 'MPICH_GNI_DEVICE', 
                   'MPICH_GNI_NDREG_ENTRIES', 'MPICH_GNI_MEMORY_REGIONS']
        
        for var in mpi_vars:
            if var in os.environ:
                print(f"{var}: {os.environ[var]}")
            else:
                print(f"{var}: Not set")
        print()
        
        # Check HPE MPT specific variables
        mpt_vars = ['MPT_DEVICE', 'MPT_NETMASK', 'MPT_GNI_DEVICE']
        for var in mpt_vars:
            if var in os.environ:
                print(f"{var}: {os.environ[var]}")
            else:
                print(f"{var}: Not set")
        print()

def check_system_limits():
    """Check system limits that might affect network performance"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== System Limits ===")
        
        # Check TCP buffer sizes
        print("TCP buffer sizes:")
        print(run_command("sysctl net.core.rmem_max net.core.wmem_max net.ipv4.tcp_rmem net.ipv4.tcp_wmem"))
        print()
        
        # Check network interface statistics
        print("Network interface statistics:")
        print(run_command("cat /proc/net/dev"))
        print()

def test_network_connectivity():
    """Test basic network connectivity between nodes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== Network Connectivity Test ===")
    
    # Get hostname
    hostname = platform.node()
    
    # Gather all hostnames
    all_hostnames = comm.gather(hostname, root=0)
    
    if rank == 0:
        unique_hosts = list(set(all_hostnames))
        print(f"Unique hosts: {unique_hosts}")
        
        # Test connectivity between hosts
        for i, host1 in enumerate(unique_hosts):
            for host2 in unique_hosts[i+1:]:
                print(f"\nTesting connectivity: {host1} -> {host2}")
                
                # Try ping (if available)
                ping_result = run_command(f"ping -c 3 {host2}")
                if "Error" not in ping_result:
                    print(f"Ping result: {ping_result.split('---')[0] if '---' in ping_result else ping_result}")
                else:
                    print("Ping not available or failed")
                
                # Try traceroute (if available)
                trace_result = run_command(f"traceroute {host2}")
                if "Error" not in trace_result:
                    print(f"Traceroute: {trace_result.split('\n')[0] if trace_result else 'No route info'}")
                else:
                    print("Traceroute not available or failed")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=== HPE MPT Network Diagnostic ===")
        print(f"MPI Library: {MPI.Get_library_version()}")
        print(f"Running on {comm.Get_size()} processes")
        print()
    
    # Run diagnostics
    check_network_interfaces()
    comm.Barrier()
    
    check_mpi_network_config()
    comm.Barrier()
    
    check_system_limits()
    comm.Barrier()
    
    test_network_connectivity()
    comm.Barrier()
    
    if rank == 0:
        print("\n=== Diagnostic Summary ===")
        print("Check the above output for:")
        print("1. Network interface speeds (should show 100G for bond1)")
        print("2. MPI environment variables")
        print("3. TCP buffer sizes")
        print("4. Network connectivity between nodes")
        print("\n✅ Network diagnostic completed!")
    
    MPI.Finalize()

if __name__ == "__main__":
    main()
