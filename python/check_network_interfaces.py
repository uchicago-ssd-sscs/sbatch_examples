#!/usr/bin/env python3

import subprocess
import os
import platform

def run_command(cmd):
    """Run a command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

def check_network_interfaces():
    """Check all available network interfaces"""
    
    print("=== NETWORK INTERFACE ANALYSIS ===")
    print(f"Hostname: {platform.node()}")
    print()
    
    # Check all network interfaces
    print("1. ALL NETWORK INTERFACES:")
    stdout, stderr, code = run_command("ip link show")
    if code == 0:
        print(stdout)
    else:
        print(f"Error: {stderr}")
    
    print("\n2. INFINIBAND INTERFACES:")
    stdout, stderr, code = run_command("ip link show | grep -i ib")
    if code == 0 and stdout.strip():
        print(stdout)
    else:
        print("No InfiniBand interfaces found")
    
    print("\n3. BOND INTERFACES:")
    stdout, stderr, code = run_command("ip link show | grep -i bond")
    if code == 0 and stdout.strip():
        print(stdout)
    else:
        print("No bond interfaces found")
    
    print("\n4. ETHERNET INTERFACES:")
    stdout, stderr, code = run_command("ip link show | grep -E '(eth|eno|ens|enp)'")
    if code == 0 and stdout.strip():
        print(stdout)
    else:
        print("No Ethernet interfaces found")
    
    print("\n5. DETAILED BOND0 INFORMATION:")
    stdout, stderr, code = run_command("ip addr show bond0")
    if code == 0:
        print(stdout)
    else:
        print("bond0 interface not found or not accessible")
    
    print("\n6. INFINIBAND DEVICE INFORMATION:")
    stdout, stderr, code = run_command("ibstat")
    if code == 0:
        print(stdout)
    else:
        print("ibstat not available or no InfiniBand devices")
    
    print("\n7. INFINIBAND VERBS INFORMATION:")
    stdout, stderr, code = run_command("ibv_devinfo")
    if code == 0:
        print(stdout)
    else:
        print("ibv_devinfo not available or no InfiniBand devices")
    
    print("\n8. NETWORK SPEED INFORMATION:")
    stdout, stderr, code = run_command("ethtool bond0 2>/dev/null || echo 'ethtool not available'")
    if code == 0:
        print(stdout)
    else:
        print("ethtool not available or bond0 not found")
    
    print("\n9. ROUTING TABLE:")
    stdout, stderr, code = run_command("ip route show")
    if code == 0:
        print(stdout)
    else:
        print(f"Error: {stderr}")
    
    print("\n10. MPI ENVIRONMENT VARIABLES:")
    mpi_vars = [
        'OMPI_MCA_btl_openib_if_include',
        'OMPI_MCA_btl',
        'OMPI_MCA_oob_tcp_if_include',
        'OMPI_MCA_btl_tcp_if_include',
        'UCX_NET_DEVICES',
        'UCX_TLS'
    ]
    
    for var in mpi_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def suggest_mpi_config():
    """Suggest MPI configuration based on available interfaces"""
    
    print("\n=== MPI CONFIGURATION SUGGESTIONS ===")
    
    # Check for InfiniBand interfaces
    stdout, stderr, code = run_command("ip link show | grep -i ib")
    has_ib = code == 0 and stdout.strip()
    
    # Check for bond0
    stdout, stderr, code = run_command("ip addr show bond0")
    has_bond0 = code == 0
    
    if has_ib:
        print("✅ InfiniBand interfaces detected!")
        print("Suggested OpenMPI configuration:")
        print("export OMPI_MCA_btl_openib_if_include=\"bond0\"")
        print("export OMPI_MCA_btl=\"^openib\"")
        print("export OMPI_MCA_btl_openib_allow_ib=1")
        print()
        print("Alternative UCX configuration:")
        print("export UCX_NET_DEVICES=\"bond0\"")
        print("export UCX_TLS=\"rc,ud\"")
        
    elif has_bond0:
        print("⚠️  bond0 found but no InfiniBand interfaces detected")
        print("bond0 might be Ethernet bonding, not InfiniBand")
        print("Check if bond0 is actually InfiniBand:")
        print("ethtool bond0")
        
    else:
        print("❌ No InfiniBand or bond0 interfaces found")
        print("Your cluster might be using Ethernet for MPI communication")
        print("This would explain the lower bandwidth results")

if __name__ == "__main__":
    check_network_interfaces()
    suggest_mpi_config()
