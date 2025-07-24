# R CPU Utilization Test

This example demonstrates a regular (non-MPI) R job that performs high CPU utilization tests using only base R functions. No additional R packages are required.

## Files

- `R_cpu_test.sh` - Slurm job submission script
- `R_cpu_test.R` - R script that performs various CPU-intensive computations
- `R_cpu_test.Rout` - Output file (generated when job runs)

## What the Test Does

The R script performs several computationally intensive tasks:

1. **Large Matrix Operations**: Creates 2000x2000 matrices and performs multiplication, inversion, and eigenvalue decomposition
2. **Statistical Computations**: Generates 1 million random numbers and performs correlation analysis and linear regression
3. **Monte Carlo Simulation**: Estimates π using Monte Carlo method with 1 million iterations
4. **Sorting and Searching**: Sorts a large vector and performs search operations
5. **String Operations**: Generates and manipulates large strings
6. **Recursive Computations**: Calculates Fibonacci numbers using recursive functions
7. **Prime Number Generation**: Finds the first 1000 prime numbers

## Job Configuration

The Slurm script requests:
- 1 task with 4 CPUs per task
- 8GB of memory
- 10 minutes maximum runtime
- Standard output/error redirection

## Usage

1. Copy the files to your working directory:
   ```bash
   cp R_cpu_test.sh R_cpu_test.R ~/your_working_directory/
   cd ~/your_working_directory/
   ```

2. Submit the job:
   ```bash
   sbatch R_cpu_test.sh
   ```

3. Monitor the job:
   ```bash
   squeue -u $USER
   ```

4. Check results:
   ```bash
   cat slurm-<jobid>.out
   cat R_cpu_test.Rout
   ```

## Customization

You can modify the job parameters in `R_cpu_test.sh`:

- **CPU cores**: Change `--cpus-per-task=4` to your desired number
- **Memory**: Change `--mem=8G` to your desired amount
- **Runtime**: Change `--time=00:10:00` to your desired time limit
- **R module**: Adjust the `module load R` line to match your system's R module name

You can also modify the computational intensity in `R_cpu_test.R`:

- **Matrix size**: Change `n <- 2000` to make matrices larger/smaller
- **Simulation iterations**: Change `n_sim <- 1000000` to adjust Monte Carlo iterations
- **Vector sizes**: Modify the size of random vectors generated
- **Fibonacci limit**: Change the upper limit for Fibonacci calculations

## Expected Output

The script will output progress messages for each test and a final summary including:
- Total execution time
- R version and platform information
- Number of CPU cores detected
- Memory usage statistics

## Notes

- This test uses only base R functions, so no package installation is required
- The recursive Fibonacci calculation is intentionally inefficient to demonstrate CPU utilization
- The script includes proper error handling and progress reporting
- Results are saved to both Slurm output files and R-specific output files 