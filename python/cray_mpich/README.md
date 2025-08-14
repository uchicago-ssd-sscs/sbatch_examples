# MPI Implementation Testing

This directory contains Python scripts for testing different MPI implementations (HPE MPT, MPICH, OpenMPI, Cray MPICH) on SLURM clusters.

## Files

- **`mpi_implementation_test.py`** - **MPI implementation detection and verification test**
- **`cray_mpich_test.py`** - General MPI functionality test (works with any MPI implementation)
- **`cray_mpich_test.slurm`** - SLURM batch script for running the tests

## Usage

### Basic Test
```bash
# Navigate to the mpi testing directory
cd python/cray_mpich

# Submit the test job
sbatch cray_mpich_test.slurm

# Or run directly with mpirun
mpirun -np 4 python mpi_implementation_test.py
```

### Setting Up MPI for Your System

**Since you don't have Cray MPICH, here are your options:**

#### Option 1: Use HPE MPT (Recommended for your system)
```bash
module load mpt/2.31
mpirun --version  # Should show HPE MPT
```

#### Option 2: Install MPI via Conda
```bash
# Install MPICH and mpi4py
conda install -c conda-forge mpich mpi4py

# Or install OpenMPI
conda install -c conda-forge openmpi mpi4py
```

#### Option 3: Use the HPE MPT Alternative
```bash
module load hmpt/2.31
```

### Verification
The `mpi_implementation_test.py` will show you which MPI implementation you're using:
- ✅ HPE MPT detected!
- ✅ MPICH detected!
- ✅ OpenMPI detected!
- ✅ Cray MPICH detected! (if available)

### Test Coverage

The `mpi_implementation_test.py` script includes:

**MPI Implementation Detection:**
- Detects HPE MPT, MPICH, OpenMPI, or Cray MPICH
- Checks for MPI environment variables
- Confirms mpirun reports the correct implementation
- Provides clear success/failure indicators

The `cray_mpich_test.py` script includes:

1. **Basic MPI Functionality**
   - MPI initialization and rank/size detection
   - Barrier synchronization
   - Node distribution analysis

2. **Point-to-Point Communication**
   - Send/receive operations
   - Non-blocking communication (Isend/Irecv)
   - Message passing validation

3. **Collective Operations**
   - Broadcast operations
   - Reduce and allreduce operations
   - Collective communication validation

4. **Performance Testing**
   - Bandwidth measurements with different data sizes
   - Send/receive timing analysis
   - Performance metrics reporting

## SLURM Configuration

The `cray_mpich_test.slurm` script is configured for:
- 2 nodes with 2 tasks per node (4 total processes)
- 10-minute time limit
- 4GB memory per job
- Automatic output/error file naming

## Expected Output

The test will display:
- MPI library version and implementation details
- Successful communication between nodes
- Performance metrics (bandwidth, timing)
- Node distribution and process mapping
- Test completion status

## Troubleshooting

If you encounter issues:

1. **Check MPI Installation**: `mpirun --version`
2. **Verify Python MPI Bindings**: `python -c "from mpi4py import MPI; print(MPI.Get_library_version())"`
3. **Load Required Modules**: Ensure Python and MPI modules are loaded
4. **Check SLURM Configuration**: Verify proper node allocation

## Advantages of Python Testing

- **Cross-Platform**: Works with any MPI implementation
- **Easy Debugging**: Python's error messages are more readable
- **Rich Ecosystem**: Access to NumPy and scientific libraries
- **Rapid Prototyping**: Quick to write and modify tests
- **Performance Analysis**: Easy to add timing and profiling
