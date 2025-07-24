#!/bin/bash
#SBATCH --job-name=comprehensive_test_mpi
#SBATCH --output=comprehensive_test_mpi_%j.out
#SBATCH --error=comprehensive_test_mpi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=dev

# Comprehensive Test Script for Multiple Programming Languages with MPI Support
# Tests: Fortran, MATLAB, Python, R, Stata (serial) + MPI Fortran, MPI C, MPI Python, MPI R, MPI MATLAB (parallel)
# MPI Features: Point-to-point, Collective communication, Non-blocking operations, Performance benchmarking
# Author: Generated for SLURM batch processing
# Date: 2025-07-24
# Author: Eric Hoy

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(pwd)/test_output"
LOG_FILE="${WORK_DIR}/comprehensive_test.log"
RESULTS_FILE="${WORK_DIR}/test_results.txt"
ERROR_LOG="${WORK_DIR}/error_details.log"

# Test status tracking
declare -A TEST_STATUS
declare -A TEST_ERRORS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize logging
init_logging() {
    # Create work directory FIRST, before any tee commands
    mkdir -p "$WORK_DIR"
    
    echo "=== Comprehensive Test Suite Started ===" | tee -a "$LOG_FILE"
    echo "Start Time: $(date)" | tee -a "$LOG_FILE"
    echo "Host: $(hostname)" | tee -a "$LOG_FILE"
    echo "User: $(whoami)" | tee -a "$LOG_FILE"
    echo "Working Directory: $(pwd)" | tee -a "$LOG_FILE"
    echo "SLURM Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
    echo "SLURM Node: $SLURM_NODELIST" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Clear previous results
    > "$RESULTS_FILE"
    > "$ERROR_LOG"
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"
    local test_description="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log_info "Starting test: $test_name - $test_description"
    echo "=== $test_name Test ===" >> "$RESULTS_FILE"
    echo "Description: $test_description" >> "$RESULTS_FILE"
    echo "Command: $test_command" >> "$RESULTS_FILE"
    echo "Start Time: $(date)" >> "$RESULTS_FILE"
    
    if eval "$test_command" >> "$RESULTS_FILE" 2>> "$ERROR_LOG"; then
        TEST_STATUS["$test_name"]="PASSED"
        TEST_ERRORS["$test_name"]=""
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_success "$test_name completed successfully"
        echo "Status: PASSED" >> "$RESULTS_FILE"
    else
        TEST_STATUS["$test_name"]="FAILED"
        TEST_ERRORS["$test_name"]="$(tail -n 10 "$ERROR_LOG" 2>/dev/null || echo 'Error details not available')"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_error "$test_name failed"
        echo "Status: FAILED" >> "$RESULTS_FILE"
        echo "Error Details:" >> "$RESULTS_FILE"
        echo "${TEST_ERRORS[$test_name]}" >> "$RESULTS_FILE"
    fi
    
    echo "End Time: $(date)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# System information gathering
gather_system_info() {
    log_info "Gathering system information..."
    
    echo "=== System Information ===" >> "$RESULTS_FILE"
    echo "OS: $(uname -a)" >> "$RESULTS_FILE"
    echo "Kernel: $(uname -r)" >> "$RESULTS_FILE"
    echo "Architecture: $(uname -m)" >> "$RESULTS_FILE"
    echo "CPU Info:" >> "$RESULTS_FILE"
    lscpu | head -20 >> "$RESULTS_FILE" 2>/dev/null || echo "lscpu not available" >> "$RESULTS_FILE"
    echo "Memory Info:" >> "$RESULTS_FILE"
    free -h >> "$RESULTS_FILE" 2>/dev/null || echo "free command not available" >> "$RESULTS_FILE"
    echo "Disk Usage:" >> "$RESULTS_FILE"
    df -h . >> "$RESULTS_FILE" 2>/dev/null || echo "df command not available" >> "$RESULTS_FILE"
    
    # MPI-specific information
    echo "MPI Information:" >> "$RESULTS_FILE"
    if command_exists mpirun; then
        echo "MPI Runtime: $(which mpirun)" >> "$RESULTS_FILE"
        mpirun --version >> "$RESULTS_FILE" 2>/dev/null || echo "mpirun version not available" >> "$RESULTS_FILE"
    fi
    
    if command_exists mpif90; then
        echo "MPI Fortran Compiler: $(which mpif90)" >> "$RESULTS_FILE"
        mpif90 --version >> "$RESULTS_FILE" 2>/dev/null || echo "mpif90 version not available" >> "$RESULTS_FILE"
    fi
    
    # SLURM MPI information
    echo "SLURM MPI Configuration:" >> "$RESULTS_FILE"
    echo "SLURM_NODELIST: $SLURM_NODELIST" >> "$RESULTS_FILE"
    echo "SLURM_NNODES: $SLURM_NNODES" >> "$RESULTS_FILE"
    echo "SLURM_NTASKS: $SLURM_NTASKS" >> "$RESULTS_FILE"
    echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" >> "$RESULTS_FILE"
    echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" >> "$RESULTS_FILE"
    
    echo "" >> "$RESULTS_FILE"
}

# Test 1: Fortran
test_fortran() {
    local test_name="Fortran"
    local test_file="${WORK_DIR}/test_fortran.f90"
    local test_exe="${WORK_DIR}/test_fortran"
    
    # Create comprehensive Fortran test program using only built-in features
    cat > "$test_file" << 'EOF'
program test_fortran
    implicit none
    
    ! Variable declarations
    integer :: i, j, sum_val, factorial
    real :: pi_approx, x, y, z
    real, dimension(5) :: array1, array2, array3
    real, dimension(3,3) :: matrix
    character(len=20) :: test_string
    logical :: test_logical
    complex :: complex_num
    
    print *, "Fortran Test Program (Built-in Features Only)"
    print *, "============================================="
    
    ! Test basic arithmetic
    sum_val = 0
    do i = 1, 10
        sum_val = sum_val + i
    end do
    print *, "Sum of 1 to 10:", sum_val
    
    ! Test factorial calculation
    factorial = 1
    do i = 1, 5
        factorial = factorial * i
    end do
    print *, "Factorial of 5:", factorial
    
    ! Test floating point operations
    pi_approx = 4.0 * atan(1.0)
    print *, "Approximation of pi:", pi_approx
    print *, "Square root of 16:", sqrt(16.0)
    print *, "Exponential of 1:", exp(1.0)
    print *, "Natural log of e:", log(exp(1.0))
    
    ! Test trigonometric functions
    x = 0.5
    print *, "sin(0.5):", sin(x)
    print *, "cos(0.5):", cos(x)
    print *, "tan(0.5):", tan(x)
    print *, "asin(0.5):", asin(x)
    print *, "acos(0.5):", acos(x)
    print *, "atan(0.5):", atan(x)
    
    ! Test array operations
    print *, "Array operations:"
    do i = 1, 5
        array1(i) = real(i)
        array2(i) = real(i) * 2.0
    end do
    
    print *, "Array1:", array1
    print *, "Array2:", array2
    
    ! Array addition
    array3 = array1 + array2
    print *, "Array1 + Array2:", array3
    
    ! Array multiplication
    array3 = array1 * array2
    print *, "Array1 * Array2:", array3
    
    ! Test matrix operations
    print *, "Matrix operations:"
    do i = 1, 3
        do j = 1, 3
            matrix(i,j) = real(i + j)
        end do
    end do
    
    print *, "Matrix:"
    do i = 1, 3
        print *, matrix(i,:)
    end do
    
    ! Test string operations
    test_string = "Hello Fortran"
    print *, "String:", test_string
    print *, "String length:", len(test_string)
    print *, "String (first 5 chars):", test_string(1:5)
    
    ! Test logical operations
    test_logical = .true.
    print *, "Logical value:", test_logical
    print *, "NOT logical:", .not. test_logical
    
    ! Test complex numbers
    complex_num = cmplx(3.0, 4.0)
    print *, "Complex number:", complex_num
    print *, "Real part:", real(complex_num)
    print *, "Imaginary part:", aimag(complex_num)
    print *, "Complex conjugate:", conjg(complex_num)
    
    ! Test random number generation
    call random_seed()
    call random_number(x)
    print *, "Random number:", x
    
    ! Test intrinsic functions
    print *, "Intrinsic functions:"
    print *, "Absolute value of -5:", abs(-5)
    print *, "Maximum of 3, 7, 2:", max(3, 7, 2)
    print *, "Minimum of 3, 7, 2:", min(3, 7, 2)
    print *, "Modulo 17 % 5:", mod(17, 5)
    
    ! Test conditional statements
    print *, "Conditional statements:"
    if (x > 0.5) then
        print *, "Random number is greater than 0.5"
    else
        print *, "Random number is less than or equal to 0.5"
    end if
    
    ! Test loops
    print *, "Loop test:"
    do i = 1, 3
        print *, "Loop iteration:", i
    end do
    
    ! Test nested loops
    print *, "Nested loop test:"
    do i = 1, 3
        do j = 1, 3
            print *, "i=", i, "j=", j, "i*j=", i*j
        end do
    end do
    
    ! Test file operations
    print *, "File operations:"
    open(unit=10, file='fortran_test.txt', status='replace')
    write(10, *) "This is a test file created by Fortran"
    write(10, *) "Line 2 of the test file"
    write(10, *) "Line 3 of the test file"
    close(10)
    print *, "Created file: fortran_test.txt"
    
    ! Read the file back
    open(unit=11, file='fortran_test.txt', status='old')
    print *, "File contents:"
    do
        read(11, *, end=100) test_string
        print *, trim(test_string)
    end do
100 continue
    close(11)
    
    ! Test formatted output
    print *, "Formatted output:"
    print '(A, F8.4)', "Pi to 4 decimal places:", pi_approx
    print '(A, I3)', "Sum in 3-digit format:", sum_val
    print '(A, E12.4)', "Pi in scientific notation:", pi_approx
    
    ! Test array functions
    print *, "Array functions:"
    print *, "Sum of array1:", sum(array1)
    print *, "Product of array1:", product(array1)
    print *, "Maximum of array1:", maxval(array1)
    print *, "Minimum of array1:", minval(array1)
    print *, "Size of array1:", size(array1)
    
    ! Test mathematical constants
    print *, "Mathematical constants:"
    print *, "Pi (approximation):", 4.0 * atan(1.0)
    print *, "e (approximation):", exp(1.0)
    
    print *, "Fortran test completed successfully!"
    
end program test_fortran
EOF
    
    local test_command="gfortran -o '$test_exe' '$test_file' && '$test_exe'"
    run_test "$test_name" "$test_command" "Compile and run a comprehensive Fortran program using only built-in features"
}

# Test 2: MATLAB
test_matlab() {
    local test_name="MATLAB"
    local test_file="${WORK_DIR}/test_matlab.m"
    
    # Create comprehensive MATLAB test script using only built-in functions
    cat > "$test_file" << 'EOF'
% MATLAB Test Script using only built-in functions
fprintf('MATLAB Test Program (Built-in Functions Only)\n');
fprintf('=============================================\n');

% Test basic arithmetic operations
a = 1:10;
sum_a = sum(a);
mean_a = mean(a);
std_a = std(a);
fprintf('Sum of 1 to 10: %d\n', sum_a);
fprintf('Mean: %.2f\n', mean_a);
fprintf('Standard deviation: %.2f\n', std_a);

% Test matrix operations
A = magic(3);
fprintf('Magic square:\n');
disp(A);
fprintf('Determinant: %.2f\n', det(A));
fprintf('Inverse:\n');
disp(inv(A));
fprintf('Eigenvalues:\n');
disp(eig(A));

% Test string operations
test_string = 'Hello MATLAB';
fprintf('String: %s\n', test_string);
fprintf('String length: %d\n', length(test_string));
fprintf('Uppercase: %s\n', upper(test_string));
fprintf('Lowercase: %s\n', lower(test_string));

% Test cell arrays
cell_array = {'apple', 'banana', 'cherry'};
fprintf('Cell array: ');
for i = 1:length(cell_array)
    fprintf('%s ', cell_array{i});
end
fprintf('\n');

% Test structures
person.name = 'John';
person.age = 30;
person.city = 'New York';
fprintf('Structure field - Name: %s, Age: %d, City: %s\n', ...
    person.name, person.age, person.city);

% Test logical operations
logical_array = [true, false, true, false];
fprintf('Logical array: ');
disp(logical_array);
fprintf('Any true: %d\n', any(logical_array));
fprintf('All true: %d\n', all(logical_array));

% Test mathematical functions
x = pi/4;
fprintf('sin(pi/4): %.4f\n', sin(x));
fprintf('cos(pi/4): %.4f\n', cos(x));
fprintf('tan(pi/4): %.4f\n', tan(x));
fprintf('sqrt(16): %.0f\n', sqrt(16));
fprintf('log(10): %.4f\n', log(10));
fprintf('exp(1): %.4f\n', exp(1));

% Test random number generation
rng(42); % Set seed for reproducible results
random_numbers = rand(1, 5);
fprintf('Random numbers: ');
fprintf('%.4f ', random_numbers);
fprintf('\n');

% Test array indexing and slicing
matrix = reshape(1:12, 3, 4);
fprintf('Original matrix:\n');
disp(matrix);
fprintf('First row: ');
fprintf('%d ', matrix(1, :));
fprintf('\n');
fprintf('Second column: ');
fprintf('%d ', matrix(:, 2));
fprintf('\n');

% Test file operations
% Create a simple text file
fid = fopen('test_file.txt', 'w');
fprintf(fid, 'Line 1\n');
fprintf(fid, 'Line 2\n');
fprintf(fid, 'Line 3\n');
fclose(fid);
fprintf('Created test_file.txt\n');

% Read the file back
fid = fopen('test_file.txt', 'r');
file_content = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
fprintf('File content:\n');
for i = 1:length(file_content{1})
    fprintf('  %s\n', file_content{1}{i});
end

% Test date and time functions
current_time = now;
fprintf('Current time: %s\n', datestr(current_time));
fprintf('Current date: %s\n', datestr(current_time, 'yyyy-mm-dd'));

% Test plotting (non-interactive) using built-in functions only
x = linspace(0, 2*pi, 100);
y = sin(x);
figure('Visible', 'off');
plot(x, y, 'b-', 'LineWidth', 2);
title('Test Plot: Sine Wave');
xlabel('x');
ylabel('sin(x)');
grid on;
saveas(gcf, 'test_plot.png');
close(gcf);
fprintf('Plot saved as test_plot.png\n');

% Test statistical functions
data = randn(1000, 1);
fprintf('Random normal data statistics:\n');
fprintf('  Mean: %.4f\n', mean(data));
fprintf('  Standard deviation: %.4f\n', std(data));
fprintf('  Median: %.4f\n', median(data));
fprintf('  Min: %.4f\n', min(data));
fprintf('  Max: %.4f\n', max(data));

% Test correlation
x_data = 1:100;
y_data = 2*x_data + randn(1, 100)*5;
correlation = corrcoef(x_data, y_data);
fprintf('Correlation between x and y: %.4f\n', correlation(1, 2));

% Test linear regression
p = polyfit(x_data, y_data, 1);
fprintf('Linear fit coefficients: y = %.2f*x + %.2f\n', p(1), p(2));

% Test control structures
fprintf('Testing control structures:\n');
for i = 1:5
    fprintf('  Loop iteration: %d\n', i);
end

% Test function handles
f = @(x) x^2 + 2*x + 1;
fprintf('Function test: f(3) = %.0f\n', f(3));

% Test array operations
array_1 = [1, 2, 3];
array_2 = [4, 5, 6];
fprintf('Array addition: ');
fprintf('%d ', array_1 + array_2);
fprintf('\n');
fprintf('Array multiplication: ');
fprintf('%d ', array_1 .* array_2);
fprintf('\n');

fprintf('MATLAB test completed successfully!\n');
exit(0);
EOF
    
    local test_command="matlab -batch \"run('$test_file')\""
    run_test "$test_name" "$test_command" "Execute a MATLAB script using only built-in functions"
}

# Test 3: Python
test_python() {
    local test_name="Python"
    local test_file="${WORK_DIR}/test_python.py"
    
    # Create comprehensive Python test script using only built-in modules
    cat > "$test_file" << 'EOF'
#!/usr/bin/env python3
"""
Python Test Script
Tests basic Python functionality using only built-in modules
"""

import sys
import os
import math
import random
import statistics
import json
import csv
import datetime
import time
import itertools
import collections

def test_basic_python():
    """Test basic Python operations"""
    print("Python Test Program")
    print("==================")
    
    # Basic arithmetic
    result = sum(range(1, 11))
    print(f"Sum of 1 to 10: {result}")
    
    # String operations
    test_string = "Hello, World!"
    print(f"String test: {test_string}")
    print(f"String length: {len(test_string)}")
    print(f"Uppercase: {test_string.upper()}")
    
    # List comprehensions
    squares = [x**2 for x in range(5)]
    print(f"Squares 0-4: {squares}")
    
    # Dictionary operations
    test_dict = {'a': 1, 'b': 2, 'c': 3}
    print(f"Dictionary: {test_dict}")
    print(f"Dictionary keys: {list(test_dict.keys())}")
    
    return True

def test_math_operations():
    """Test mathematical operations using built-in math module"""
    print("\nMath Operations:")
    
    # Basic math functions
    print(f"Pi approximation: {math.pi}")
    print(f"Square root of 16: {math.sqrt(16)}")
    print(f"Sine of pi/2: {math.sin(math.pi/2)}")
    print(f"Cosine of 0: {math.cos(0)}")
    print(f"Natural log of e: {math.log(math.e)}")
    print(f"e^2: {math.exp(2)}")
    
    # Random number generation
    random.seed(42)  # For reproducible results
    random_numbers = [random.random() for _ in range(5)]
    print(f"Random numbers: {random_numbers}")
    
    # Statistics using built-in statistics module
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Data: {data}")
    print(f"Mean: {statistics.mean(data)}")
    print(f"Median: {statistics.median(data)}")
    print(f"Standard deviation: {statistics.stdev(data)}")
    
    return True

def test_file_operations():
    """Test file operations using built-in modules"""
    print("\nFile Operations:")
    
    # Create a simple CSV file
    csv_file = "test_data.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Age', 'City'])
        writer.writerow(['Alice', 25, 'New York'])
        writer.writerow(['Bob', 30, 'London'])
        writer.writerow(['Charlie', 35, 'Paris'])
    
    print(f"Created CSV file: {csv_file}")
    
    # Read and display CSV content
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(f"CSV row: {row}")
    
    # Create a JSON file
    json_data = {
        'test': True,
        'numbers': [1, 2, 3, 4, 5],
        'string': 'Hello JSON',
        'nested': {'key': 'value'}
    }
    
    json_file = "test_data.json"
    with open(json_file, 'w') as file:
        json.dump(json_data, file, indent=2)
    
    print(f"Created JSON file: {json_file}")
    
    # Read and display JSON content
    with open(json_file, 'r') as file:
        loaded_data = json.load(file)
        print(f"Loaded JSON: {loaded_data}")
    
    return True

def test_collections():
    """Test collections module"""
    print("\nCollections Module:")
    
    # Counter
    from collections import Counter
    words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
    word_count = Counter(words)
    print(f"Word count: {word_count}")
    
    # DefaultDict
    from collections import defaultdict
    dd = defaultdict(list)
    for i in range(5):
        dd[i % 2].append(i)
    print(f"DefaultDict: {dict(dd)}")
    
    # NamedTuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(11, 22)
    print(f"NamedTuple Point: {p}")
    print(f"Point x: {p.x}, y: {p.y}")
    
    return True

def test_itertools():
    """Test itertools module"""
    print("\nItertools Module:")
    
    # Combinations
    items = ['A', 'B', 'C']
    combos = list(itertools.combinations(items, 2))
    print(f"Combinations of {items} (2): {combos}")
    
    # Permutations
    perms = list(itertools.permutations(items, 2))
    print(f"Permutations of {items} (2): {perms}")
    
    # Infinite iterator
    counter = itertools.count(1)
    first_five = [next(counter) for _ in range(5)]
    print(f"First 5 from counter: {first_five}")
    
    return True

def test_datetime():
    """Test datetime operations"""
    print("\nDateTime Operations:")
    
    # Current time
    now = datetime.datetime.now()
    print(f"Current time: {now}")
    print(f"Current date: {now.date()}")
    print(f"Current time: {now.time()}")
    
    # Time formatting
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Formatted time: {formatted}")
    
    # Time arithmetic
    future = now + datetime.timedelta(days=7)
    print(f"One week from now: {future}")
    
    return True

def main():
    """Main test function"""
    try:
        test_basic_python()
        test_math_operations()
        test_file_operations()
        test_collections()
        test_itertools()
        test_datetime()
        
        print("\nPython test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Python test failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    local test_command="python3 '$test_file'"
    run_test "$test_name" "$test_command" "Execute a comprehensive Python script using only built-in modules"
}

# Test 4: R
test_r() {
    local test_name="R"
    local test_file="${WORK_DIR}/test_r.R"
    
    # Create comprehensive R test script using only base R functions
    cat > "$test_file" << 'EOF'
# R Test Script using only base R functions
cat("R Test Program (Base R Only)\n")
cat("===========================\n")

# Test basic R operations
result <- sum(1:10)
cat("Sum of 1 to 10:", result, "\n")

# Test vector operations
vec <- c(1, 2, 3, 4, 5)
cat("Vector:", vec, "\n")
cat("Mean:", mean(vec), "\n")
cat("Standard deviation:", sd(vec), "\n")
cat("Median:", median(vec), "\n")
cat("Min:", min(vec), "Max:", max(vec), "\n")

# Test string operations
strings <- c("Hello", "World", "R", "Programming")
cat("Strings:", strings, "\n")
cat("String length:", nchar(strings), "\n")
cat("Uppercase:", toupper(strings), "\n")
cat("Substring:", substr(strings, 1, 3), "\n")

# Test matrix operations
mat <- matrix(c(2, 1, 1, 3, 2, 1, 1, 1, 2), nrow=3, ncol=3)
cat("Matrix:\n")
print(mat)
cat("Matrix determinant:", det(mat), "\n")
cat("Matrix transpose:\n")
print(t(mat))
cat("Matrix inverse:\n")
print(solve(mat))

# Test data frame operations
df <- data.frame(
  Name = c("Alice", "Bob", "Charlie"),
  Age = c(25, 30, 35),
  City = c("New York", "London", "Paris")
)
cat("Data frame:\n")
print(df)
cat("Data frame structure:\n")
str(df)
cat("Data frame summary:\n")
print(summary(df))

# Test list operations
my_list <- list(
  numbers = 1:5,
  strings = c("a", "b", "c"),
  matrix = matrix(1:4, 2, 2)
)
cat("List:\n")
print(my_list)
cat("List names:", names(my_list), "\n")

# Test factor operations
colors <- factor(c("red", "blue", "red", "green", "blue"))
cat("Factor:", colors, "\n")
cat("Factor levels:", levels(colors), "\n")
cat("Factor summary:\n")
print(table(colors))

# Test date and time operations
current_date <- Sys.Date()
current_time <- Sys.time()
cat("Current date:", current_date, "\n")
cat("Current time:", current_time, "\n")
cat("Date format:", format(current_date, "%Y-%m-%d"), "\n")

# Test file operations
# Create a simple text file
writeLines(c("Line 1", "Line 2", "Line 3"), "test_file.txt")
cat("Created test_file.txt\n")

# Read the file back
file_content <- readLines("test_file.txt")
cat("File content:", file_content, "\n")

# Test plotting (non-interactive) using base R only
pdf("test_plot.pdf")
x <- seq(0, 2*pi, length.out=100)
y <- sin(x)
plot(x, y, type="l", col="blue", lwd=2, 
     main="Test Plot: Sine Wave", 
     xlab="x", ylab="sin(x)")
abline(h=0, lty=2)
grid()
dev.off()

cat("Plot saved as test_plot.pdf\n")

# Test statistical functions
set.seed(123)  # For reproducible results
data <- rnorm(1000, mean=0, sd=1)
cat("Random normal data summary:\n")
print(summary(data))
cat("Data variance:", var(data), "\n")
cat("Data quantiles:\n")
print(quantile(data, probs=c(0.25, 0.5, 0.75)))

# Test correlation
x_data <- 1:100
y_data <- 2*x_data + rnorm(100, 0, 5)
correlation <- cor(x_data, y_data)
cat("Correlation between x and y:", correlation, "\n")

# Test linear regression
lm_result <- lm(y_data ~ x_data)
cat("Linear regression summary:\n")
print(summary(lm_result))

# Test control structures
cat("Testing control structures:\n")
for(i in 1:5) {
  cat("Loop iteration:", i, "\n")
}

# Test function definition
my_function <- function(x) {
  return(x^2 + 2*x + 1)
}
cat("Function test: my_function(3) =", my_function(3), "\n")

# Test apply functions
matrix_2d <- matrix(1:12, 3, 4)
cat("Matrix for apply test:\n")
print(matrix_2d)
cat("Row sums:", apply(matrix_2d, 1, sum), "\n")
cat("Column means:", apply(matrix_2d, 2, mean), "\n")

cat("R test completed successfully!\n")
quit(status=0)
EOF
    
    local test_command="Rscript '$test_file'"
    run_test "$test_name" "$test_command" "Execute an R script using only base R functions"
}

# Test 5: Stata
test_stata() {
    local test_name="Stata"
    local test_file="${WORK_DIR}/test_stata.do"
    
    # Create comprehensive Stata test script using only built-in commands
    cat > "$test_file" << 'EOF'
* Stata Test Script using only built-in commands
display "Stata Test Program (Built-in Commands Only)"
display "============================================"

* Test basic operations and macros
local sum = 0
forvalues i = 1/10 {
    local sum = `sum' + `i'
}
display "Sum of 1 to 10: `sum'"

* Test string operations
local test_string = "Hello Stata"
display "String: `test_string'"
display "String length: " + strlen("`test_string'")

* Test data generation
clear
set obs 100
display "Created dataset with 100 observations"

* Generate variables
generate x = rnormal(0, 1)
generate y = 2*x + rnormal(0, 0.5)
generate z = rpoisson(5)
generate category = mod(_n, 3) + 1
label define cat_lbl 1 "Low" 2 "Medium" 3 "High"
label values category cat_lbl

display "Generated variables: x, y, z, category"

* Test descriptive statistics
display "Descriptive statistics for all variables:"
summarize

display "Detailed statistics for x:"
summarize x, detail

display "Summary by category:"
tabstat x y z, by(category) statistics(mean sd min max) columns(statistics)

* Test frequency tables
display "Frequency table for category:"
tabulate category

* Test correlation
display "Correlation matrix:"
correlate x y z

* Test regression analysis
display "Linear regression of y on x:"
regress y x

display "Regression with robust standard errors:"
regress y x, robust

* Test hypothesis testing
display "Testing if mean of x equals 0:"
ttest x == 0

* Test plotting capabilities
display "Creating scatter plot..."
twoway scatter y x, title("Test Plot: Y vs X") ///
    xtitle("X") ytitle("Y") ///
    name(test_plot, replace)

* Save plot
graph export test_plot.png, replace
display "Plot saved as test_plot.png"

* Test histogram
display "Creating histogram..."
histogram x, title("Histogram of X") ///
    xtitle("X") ytitle("Frequency") ///
    name(hist_plot, replace)
graph export hist_plot.png, replace
display "Histogram saved as hist_plot.png"

* Test box plot
display "Creating box plot..."
graph box x, title("Box Plot of X") ///
    ytitle("X") ///
    name(box_plot, replace)
graph export box_plot.png, replace
display "Box plot saved as box_plot.png"

* Test data manipulation
display "Creating new variables:"
generate x_squared = x^2
generate log_y = log(abs(y) + 1)
generate interaction = x * z

display "New variables created: x_squared, log_y, interaction"

* Test sorting and listing
display "First 10 observations sorted by x:"
sort x
list x y z in 1/10

* Test subsetting
display "Statistics for observations where x > 0:"
summarize if x > 0

* Test data export
display "Exporting data to CSV:"
export delimited using "test_data.csv", replace
display "Data exported to test_data.csv"

* Test data import
display "Importing data back:"
import delimited using "test_data.csv", clear
display "Data imported successfully"

* Test matrix operations
display "Matrix operations:"
matrix A = (1, 2 \ 3, 4)
matrix B = (5, 6 \ 7, 8)
matrix C = A + B
matrix list A
matrix list B
matrix list C

* Test programming features
display "Testing programming features:"
forvalues i = 1/5 {
    display "Loop iteration `i'"
}

* Test conditional logic
display "Testing conditional logic:"
if 1 {
    display "Condition is true"
}
else {
    display "Condition is false"
}

* Test function-like behavior with programs
capture program drop my_program
program define my_program
    display "Hello from my_program"
    display "Argument 1: `1'"
end

my_program "test argument"
display "Program executed successfully"

* Test file operations
display "Creating test file:"
file open testfile using "test_output.txt", write replace
file write testfile "This is a test file created by Stata" _newline
file write testfile "Line 2 of the test file" _newline
file close testfile
display "Test file created: test_output.txt"

* Test date and time
display "Current date and time:"
display c(current_date)
display c(current_time)

* Test system information
display "System information:"
display "Stata version: " + c(stata_version)
display "Operating system: " + c(os)

display "Stata test completed successfully!"
exit, clear
EOF
    
    local test_command="stata -b do '$test_file'"
    run_test "$test_name" "$test_command" "Execute a Stata script using only built-in commands"
}

# Test 6: MPI Fortran
test_mpi_fortran() {
    local test_name="MPI_Fortran"
    local test_file="${WORK_DIR}/test_mpi_fortran.f90"
    local test_exe="${WORK_DIR}/test_mpi_fortran"
    
    # Create comprehensive MPI Fortran test program
    cat > "$test_file" << 'EOF'
program test_mpi_fortran
    use mpi
    implicit none
    
    integer :: rank, size, ierr, i, j
    integer :: tag, source, dest, count
    integer, dimension(MPI_STATUS_SIZE) :: status
    real :: start_time, end_time, local_sum, global_sum
    real, dimension(100) :: local_data, received_data
    character(len=100) :: message
    
    ! Initialize MPI
    call MPI_Init(ierr)
    if (ierr /= MPI_SUCCESS) then
        print *, "MPI_Init failed"
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    
    ! Get process rank and size
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    if (rank == 0) then
        print *, "MPI Fortran Test Program"
        print *, "======================="
        print *, "Number of processes:", size
        print *, ""
    end if
    
    ! Synchronize all processes
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    ! Test 1: Basic communication - Hello World
    write(message, '(A,I3,A,I3)') "Hello from process ", rank, " of ", size
    print *, trim(message)
    
    ! Test 2: Point-to-point communication
    if (rank == 0) then
        ! Master process sends data to all other processes
        do dest = 1, size-1
            local_data(1) = real(dest) * 10.0
            call MPI_Send(local_data, 1, MPI_REAL, dest, 100, MPI_COMM_WORLD, ierr)
            print *, "Sent", local_data(1), "to process", dest
        end do
    else
        ! Worker processes receive data from master
        call MPI_Recv(received_data, 1, MPI_REAL, 0, 100, MPI_COMM_WORLD, status, ierr)
        print *, "Process", rank, "received", received_data(1)
    end if
    
    ! Test 3: Collective communication - Broadcast
    if (rank == 0) then
        local_data(1) = 42.0
        print *, "Broadcasting value", local_data(1), "to all processes"
    end if
    
    call MPI_Bcast(local_data, 1, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
    print *, "Process", rank, "received broadcast:", local_data(1)
    
    ! Test 4: Collective communication - Reduce
    local_sum = real(rank + 1)
    call MPI_Reduce(local_sum, global_sum, 1, MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    
    if (rank == 0) then
        print *, "Sum of all process ranks + 1:", global_sum
    end if
    
    ! Test 5: Collective communication - Allreduce
    local_sum = real(rank * 2)
    call MPI_Allreduce(local_sum, global_sum, 1, MPI_REAL, MPI_SUM, MPI_COMM_WORLD, ierr)
    print *, "Process", rank, "Allreduce result:", global_sum
    
    ! Test 6: Collective communication - Gather
    local_data(1) = real(rank)
    call MPI_Gather(local_data, 1, MPI_REAL, received_data, 1, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
    
    if (rank == 0) then
        print *, "Gathered data from all processes:"
        do i = 1, size
            print *, "Process", i-1, ":", received_data(i)
        end do
    end if
    
    ! Test 7: Collective communication - Scatter
    if (rank == 0) then
        do i = 1, size
            local_data(i) = real(i) * 100.0
        end do
        print *, "Scattering data to all processes"
    end if
    
    call MPI_Scatter(local_data, 1, MPI_REAL, received_data, 1, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
    print *, "Process", rank, "received scattered data:", received_data(1)
    
    ! Test 8: Non-blocking communication
    if (rank == 0) then
        do dest = 1, size-1
            local_data(1) = real(dest) * 1000.0
            call MPI_Isend(local_data, 1, MPI_REAL, dest, 200, MPI_COMM_WORLD, MPI_REQUEST_NULL, ierr)
        end do
    else
        call MPI_Irecv(received_data, 1, MPI_REAL, 0, 200, MPI_COMM_WORLD, MPI_REQUEST_NULL, ierr)
    end if
    
    ! Test 9: Timing and performance
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    start_time = MPI_Wtime()
    
    ! Do some work
    do i = 1, 1000000
        local_sum = local_sum + sin(real(i) * 0.001)
    end do
    
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    end_time = MPI_Wtime()
    
    if (rank == 0) then
        print *, "Work completed in", end_time - start_time, "seconds"
    end if
    
    ! Test 10: Error handling
    if (rank == 0) then
        print *, "Testing error handling..."
        ! This should fail gracefully
        call MPI_Send(local_data, 1, MPI_REAL, size+1, 300, MPI_COMM_WORLD, ierr)
        if (ierr /= MPI_SUCCESS) then
            print *, "Expected error caught: Invalid destination rank"
        end if
    end if
    
    ! Final synchronization
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    
    if (rank == 0) then
        print *, "MPI Fortran test completed successfully!"
    end if
    
    ! Finalize MPI
    call MPI_Finalize(ierr)
    
end program test_mpi_fortran
EOF
    
    local test_command="mpif90 -o '$test_exe' '$test_file' && mpirun -np 8 '$test_exe'"
    run_test "$test_name" "$test_command" "Compile and run MPI Fortran program with 8 processes"
}

# Test 7: MPI Python
test_mpi_python() {
    local test_name="MPI_Python"
    local test_file="${WORK_DIR}/test_mpi_python.py"
    
    # Create comprehensive MPI Python test script
    cat > "$test_file" << 'EOF'
#!/usr/bin/env python3
"""
MPI Python Test Script
Tests MPI functionality using mpi4py
"""

import sys
import numpy as np
from mpi4py import MPI
import time

def test_basic_mpi():
    """Test basic MPI operations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("MPI Python Test Program")
        print("======================")
        print(f"Number of processes: {size}")
        print()
    
    # Test 1: Basic communication
    message = f"Hello from process {rank} of {size}"
    print(message)
    
    # Synchronize all processes
    comm.Barrier()
    
    return comm, rank, size

def test_point_to_point(comm, rank, size):
    """Test point-to-point communication"""
    if rank == 0:
        print("Testing point-to-point communication...")
    
    # Test send/receive
    if rank == 0:
        # Master sends data to all workers
        for dest in range(1, size):
            data = np.array([dest * 10.0], dtype=np.float64)
            comm.Send(data, dest=dest, tag=100)
            print(f"Sent {data[0]} to process {dest}")
    else:
        # Workers receive data from master
        data = np.array([0.0], dtype=np.float64)
        comm.Recv(data, source=0, tag=100)
        print(f"Process {rank} received {data[0]}")
    
    comm.Barrier()

def test_collective_communication(comm, rank, size):
    """Test collective communication operations"""
    if rank == 0:
        print("Testing collective communication...")
    
    # Test broadcast
    if rank == 0:
        data = np.array([42.0], dtype=np.float64)
        print(f"Broadcasting value {data[0]} to all processes")
    
    data = np.array([0.0], dtype=np.float64)
    comm.Bcast(data, root=0)
    print(f"Process {rank} received broadcast: {data[0]}")
    
    # Test reduce
    local_sum = np.array([rank + 1.0], dtype=np.float64)
    global_sum = np.array([0.0], dtype=np.float64)
    comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Sum of all process ranks + 1: {global_sum[0]}")
    
    # Test allreduce
    local_sum = np.array([rank * 2.0], dtype=np.float64)
    global_sum = np.array([0.0], dtype=np.float64)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    print(f"Process {rank} Allreduce result: {global_sum[0]}")
    
    # Test gather
    local_data = np.array([rank], dtype=np.float64)
    if rank == 0:
        gathered_data = np.zeros(size, dtype=np.float64)
    else:
        gathered_data = None
    
    comm.Gather(local_data, gathered_data, root=0)
    
    if rank == 0:
        print("Gathered data from all processes:")
        for i in range(size):
            print(f"  Process {i}: {gathered_data[i]}")
    
    # Test scatter
    if rank == 0:
        scatter_data = np.array([(i + 1) * 100.0 for i in range(size)], dtype=np.float64)
        print("Scattering data to all processes")
    else:
        scatter_data = None
    
    local_data = np.array([0.0], dtype=np.float64)
    comm.Scatter(scatter_data, local_data, root=0)
    print(f"Process {rank} received scattered data: {local_data[0]}")
    
    comm.Barrier()

def test_non_blocking_communication(comm, rank, size):
    """Test non-blocking communication"""
    if rank == 0:
        print("Testing non-blocking communication...")
    
    # Test non-blocking send/receive
    if rank == 0:
        for dest in range(1, size):
            data = np.array([dest * 1000.0], dtype=np.float64)
            req = comm.Isend(data, dest=dest, tag=200)
            req.Wait()
    else:
        data = np.array([0.0], dtype=np.float64)
        req = comm.Irecv(data, source=0, tag=200)
        req.Wait()
        print(f"Process {rank} received non-blocking data: {data[0]}")
    
    comm.Barrier()

def test_performance(comm, rank, size):
    """Test performance and timing"""
    if rank == 0:
        print("Testing performance...")
    
    comm.Barrier()
    start_time = time.time()
    
    # Do some computational work
    local_sum = 0.0
    for i in range(1000000):
        local_sum += np.sin(i * 0.001)
    
    comm.Barrier()
    end_time = time.time()
    
    if rank == 0:
        print(f"Work completed in {end_time - start_time:.4f} seconds")

def test_array_operations(comm, rank, size):
    """Test array operations with MPI"""
    if rank == 0:
        print("Testing array operations...")
    
    # Create local arrays
    local_array = np.random.random(100)
    
    # Test array reduce
    global_sum = np.zeros(100, dtype=np.float64)
    comm.Reduce(local_array, global_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Array sum computed, first 5 elements: {global_sum[:5]}")
    
    # Test array broadcast
    if rank == 0:
        broadcast_array = np.random.random(50)
    else:
        broadcast_array = np.zeros(50, dtype=np.float64)
    
    comm.Bcast(broadcast_array, root=0)
    print(f"Process {rank} received broadcast array, first 3 elements: {broadcast_array[:3]}")
    
    comm.Barrier()

def main():
    """Main test function"""
    try:
        # Initialize MPI
        comm, rank, size = test_basic_mpi()
        
        # Run MPI tests
        test_point_to_point(comm, rank, size)
        test_collective_communication(comm, rank, size)
        test_non_blocking_communication(comm, rank, size)
        test_performance(comm, rank, size)
        test_array_operations(comm, rank, size)
        
        if rank == 0:
            print("MPI Python test completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"MPI Python test failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    local test_command="mpirun -np 8 python3 '$test_file'"
    run_test "$test_name" "$test_command" "Execute MPI Python script with 8 processes using mpi4py"
}

# Test 8: MPI R
test_mpi_r() {
    local test_name="MPI_R"
    local test_file="${WORK_DIR}/test_mpi_r.R"
    
    # Create comprehensive MPI R test script
    cat > "$test_file" << 'EOF'
# MPI R Test Script using Rmpi
library(Rmpi)

# Initialize MPI
mpi.spawn.Rslaves(nslaves = 7)  # Spawn 7 slaves (8 total processes)

# Test basic MPI operations
cat("MPI R Test Program\n")
cat("==================\n")

# Get process information
rank <- mpi.comm.rank()
size <- mpi.comm.size()
cat("Process", rank, "of", size, "\n")

# Test 1: Basic communication
if (rank == 0) {
    cat("Testing basic communication...\n")
}

# Test 2: Broadcast
if (rank == 0) {
    data <- 42
    cat("Broadcasting value", data, "to all processes\n")
} else {
    data <- 0
}

data <- mpi.bcast(data, type = 1, rank = 0, comm = 1)
cat("Process", rank, "received broadcast:", data, "\n")

# Test 3: Reduce
local_sum <- rank + 1
global_sum <- mpi.reduce(local_sum, type = 2, op = "sum", rank = 0, comm = 1)

if (rank == 0) {
    cat("Sum of all process ranks + 1:", global_sum, "\n")
}

# Test 4: Allreduce
local_sum <- rank * 2
global_sum <- mpi.allreduce(local_sum, type = 2, op = "sum", comm = 1)
cat("Process", rank, "Allreduce result:", global_sum, "\n")

# Test 5: Gather
local_data <- rank
if (rank == 0) {
    gathered_data <- mpi.gather(local_data, type = 1, rank = 0, comm = 1)
    cat("Gathered data from all processes:", gathered_data, "\n")
} else {
    mpi.gather(local_data, type = 1, rank = 0, comm = 1)
}

# Test 6: Scatter
if (rank == 0) {
    scatter_data <- (1:size) * 100
    cat("Scattering data to all processes\n")
} else {
    scatter_data <- NULL
}

local_data <- mpi.scatter(scatter_data, type = 2, rank = 0, comm = 1)
cat("Process", rank, "received scattered data:", local_data, "\n")

# Test 7: Point-to-point communication
if (rank == 0) {
    for (dest in 1:(size-1)) {
        data <- dest * 10
        mpi.send(data, type = 2, rank = dest, tag = 100, comm = 1)
        cat("Sent", data, "to process", dest, "\n")
    }
} else {
    data <- mpi.recv(type = 2, rank = 0, tag = 100, comm = 1)
    cat("Process", rank, "received", data, "\n")
}

# Test 8: Array operations
if (rank == 0) {
    cat("Testing array operations...\n")
}

# Create local arrays
local_array <- runif(10)
cat("Process", rank, "local array (first 3):", local_array[1:3], "\n")

# Test array reduce
if (rank == 0) {
    global_sum <- mpi.reduce(local_array, type = 2, op = "sum", rank = 0, comm = 1)
    cat("Array sum (first 3):", global_sum[1:3], "\n")
} else {
    mpi.reduce(local_array, type = 2, op = "sum", rank = 0, comm = 1)
}

# Test 9: Performance timing
if (rank == 0) {
    cat("Testing performance...\n")
}

start_time <- Sys.time()

# Do some computational work
local_sum <- 0
for (i in 1:100000) {
    local_sum <- local_sum + sin(i * 0.001)
}

end_time <- Sys.time()
cat("Process", rank, "work completed in", as.numeric(end_time - start_time), "seconds\n")

# Test 10: Statistical operations
if (rank == 0) {
    cat("Testing statistical operations...\n")
}

# Generate local data
local_data <- rnorm(1000, mean = rank, sd = 1)

# Compute local statistics
local_mean <- mean(local_data)
local_sd <- sd(local_data)

# Gather statistics to master
if (rank == 0) {
    all_means <- mpi.gather(local_mean, type = 2, rank = 0, comm = 1)
    all_sds <- mpi.gather(local_sd, type = 2, rank = 0, comm = 1)
    
    cat("Means from all processes:", all_means, "\n")
    cat("Standard deviations from all processes:", all_sds, "\n")
} else {
    mpi.gather(local_mean, type = 2, rank = 0, comm = 1)
    mpi.gather(local_sd, type = 2, rank = 0, comm = 1)
}

# Final synchronization
mpi.barrier(comm = 1)

if (rank == 0) {
    cat("MPI R test completed successfully!\n")
}

# Clean up
mpi.close.Rslaves()
mpi.quit()
EOF
    
    local test_command="mpirun -np 8 Rscript '$test_file'"
    run_test "$test_name" "$test_command" "Execute MPI R script with 8 processes using Rmpi"
}

# Test 9: MPI MATLAB
test_mpi_matlab() {
    local test_name="MPI_MATLAB"
    local test_file="${WORK_DIR}/test_mpi_matlab.m"
    
    # Create comprehensive MPI MATLAB test script
    cat > "$test_file" << 'EOF'
% MPI MATLAB Test Script using Parallel Computing Toolbox
fprintf('MPI MATLAB Test Program\n');
fprintf('======================\n');

% Initialize parallel pool
if isempty(gcp('nocreate'))
    parpool('local', 8);
end

poolobj = gcp;
num_workers = poolobj.NumWorkers;
fprintf('Number of workers: %d\n', num_workers);

% Test 1: Basic parallel operations
fprintf('Testing basic parallel operations...\n');

% Parallel for loop
parfor i = 1:10
    fprintf('Worker processing iteration %d\n', i);
end

% Test 2: Parallel array operations
fprintf('Testing parallel array operations...\n');

% Create distributed array
d = distributed.rand(1000, 1000);
fprintf('Created distributed array of size %dx%d\n', size(d));

% Perform operations on distributed array
result = sum(d, 'all');
fprintf('Sum of distributed array: %.4f\n', result);

% Test 3: Parallel reduction
fprintf('Testing parallel reduction...\n');

% Each worker computes local sum
local_data = rand(1000, 1);
local_sum = sum(local_data);

% Reduce to get global sum
global_sum = gplus(local_sum);
fprintf('Global sum from all workers: %.4f\n', global_sum);

% Test 4: Parallel matrix operations
fprintf('Testing parallel matrix operations...\n');

% Create distributed matrices
A = distributed.rand(500, 500);
B = distributed.rand(500, 500);

% Matrix multiplication
C = A * B;
fprintf('Matrix multiplication completed\n');

% Test 5: Parallel statistical operations
fprintf('Testing parallel statistical operations...\n');

% Generate distributed random data
data = distributed.randn(10000, 100);

% Compute statistics
mean_val = mean(data, 'all');
std_val = std(data, 0, 'all');
fprintf('Mean: %.4f, Std: %.4f\n', mean_val, std_val);

% Test 6: Parallel file operations
fprintf('Testing parallel file operations...\n');

% Each worker writes to a file
spmd
    filename = sprintf('worker_%d_data.txt', labindex);
    fid = fopen(filename, 'w');
    fprintf(fid, 'Data from worker %d\n', labindex);
    for i = 1:10
        fprintf(fid, 'Line %d: %f\n', i, rand());
    end
    fclose(fid);
end

fprintf('Files created by each worker\n');

% Test 7: Parallel optimization
fprintf('Testing parallel optimization...\n');

% Define objective function
objfun = @(x) sum(x.^2);

% Parallel optimization
options = optimoptions('fmincon', 'UseParallel', true);
x0 = rand(10, 1);
[x, fval] = fmincon(objfun, x0, [], [], [], [], [], [], [], options);

fprintf('Optimization result: f(x) = %.6f\n', fval);

% Test 8: Parallel Monte Carlo
fprintf('Testing parallel Monte Carlo...\n');

% Monte Carlo integration
n_points = 1000000;
points_in_circle = 0;

parfor i = 1:n_points
    x = rand() * 2 - 1;
    y = rand() * 2 - 1;
    if x^2 + y^2 <= 1
        points_in_circle = points_in_circle + 1;
    end
end

pi_estimate = 4 * points_in_circle / n_points;
fprintf('Monte Carlo pi estimate: %.6f (error: %.6f)\n', pi_estimate, abs(pi_estimate - pi));

% Test 9: Parallel data processing
fprintf('Testing parallel data processing...\n');

% Create sample data
data_size = 100000;
sample_data = randn(data_size, 5);

% Process data in parallel
parfor i = 1:data_size
    % Simulate some processing
    processed_data(i) = sum(sample_data(i, :).^2);
end

fprintf('Processed %d data points in parallel\n', data_size);

% Test 10: Performance benchmarking
fprintf('Testing performance...\n');

% Benchmark parallel vs serial
n = 1000;
A = rand(n, n);
B = rand(n, n);

% Serial matrix multiplication
tic;
C_serial = A * B;
serial_time = toc;
fprintf('Serial matrix multiplication: %.4f seconds\n', serial_time);

% Parallel matrix multiplication
tic;
C_parallel = distributed(A) * distributed(B);
parallel_time = toc;
fprintf('Parallel matrix multiplication: %.4f seconds\n', parallel_time);
fprintf('Speedup: %.2fx\n', serial_time / parallel_time);

% Clean up
delete(gcp('nocreate'));

fprintf('MPI MATLAB test completed successfully!\n');
exit(0);
EOF
    
    local test_command="matlab -batch \"run('$test_file')\""
    run_test "$test_name" "$test_command" "Execute MPI MATLAB script using Parallel Computing Toolbox"
}

# Test 10: MPI C
test_mpi_c() {
    local test_name="MPI_C"
    local test_file="${WORK_DIR}/test_mpi_c.c"
    local test_exe="${WORK_DIR}/test_mpi_c"
    
    # Create comprehensive MPI C test program
    cat > "$test_file" << 'EOF'
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size, i, j;
    int tag = 100;
    MPI_Status status;
    double start_time, end_time;
    double local_sum, global_sum;
    double local_data[100], received_data[100];
    char message[100];
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("MPI C Test Program\n");
        printf("==================\n");
        printf("Number of processes: %d\n\n", size);
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Test 1: Basic communication - Hello World
    sprintf(message, "Hello from process %d of %d", rank, size);
    printf("%s\n", message);
    
    // Test 2: Point-to-point communication
    if (rank == 0) {
        // Master process sends data to all other processes
        for (int dest = 1; dest < size; dest++) {
            local_data[0] = (double)dest * 10.0;
            MPI_Send(local_data, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            printf("Sent %.1f to process %d\n", local_data[0], dest);
        }
    } else {
        // Worker processes receive data from master
        MPI_Recv(received_data, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        printf("Process %d received %.1f\n", rank, received_data[0]);
    }
    
    // Test 3: Collective communication - Broadcast
    if (rank == 0) {
        local_data[0] = 42.0;
        printf("Broadcasting value %.1f to all processes\n", local_data[0]);
    }
    
    MPI_Bcast(local_data, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("Process %d received broadcast: %.1f\n", rank, local_data[0]);
    
    // Test 4: Collective communication - Reduce
    local_sum = (double)(rank + 1);
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sum of all process ranks + 1: %.1f\n", global_sum);
    }
    
    // Test 5: Collective communication - Allreduce
    local_sum = (double)(rank * 2);
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Process %d Allreduce result: %.1f\n", rank, global_sum);
    
    // Test 6: Collective communication - Gather
    local_data[0] = (double)rank;
    if (rank == 0) {
        MPI_Gather(local_data, 1, MPI_DOUBLE, received_data, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("Gathered data from all processes:\n");
        for (i = 0; i < size; i++) {
            printf("  Process %d: %.1f\n", i, received_data[i]);
        }
    } else {
        MPI_Gather(local_data, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Test 7: Collective communication - Scatter
    if (rank == 0) {
        for (i = 0; i < size; i++) {
            local_data[i] = (double)(i + 1) * 100.0;
        }
        printf("Scattering data to all processes\n");
    }
    
    MPI_Scatter(local_data, 1, MPI_DOUBLE, received_data, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("Process %d received scattered data: %.1f\n", rank, received_data[0]);
    
    // Test 8: Non-blocking communication
    MPI_Request request;
    if (rank == 0) {
        for (int dest = 1; dest < size; dest++) {
            local_data[0] = (double)dest * 1000.0;
            MPI_Isend(local_data, 1, MPI_DOUBLE, dest, 200, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
    } else {
        MPI_Irecv(received_data, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        printf("Process %d received non-blocking data: %.1f\n", rank, received_data[0]);
    }
    
    // Test 9: Timing and performance
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Do some computational work
    local_sum = 0.0;
    for (i = 0; i < 1000000; i++) {
        local_sum += sin((double)i * 0.001);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Work completed in %.4f seconds\n", end_time - start_time);
    }
    
    // Test 10: Array operations
    if (rank == 0) {
        printf("Testing array operations...\n");
    }
    
    // Create local arrays
    for (i = 0; i < 100; i++) {
        local_data[i] = (double)rand() / RAND_MAX;
    }
    
    // Test array reduce
    if (rank == 0) {
        MPI_Reduce(local_data, received_data, 100, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Array sum computed, first 5 elements: %.4f %.4f %.4f %.4f %.4f\n",
               received_data[0], received_data[1], received_data[2], received_data[3], received_data[4]);
    } else {
        MPI_Reduce(local_data, NULL, 100, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    // Test array broadcast
    if (rank == 0) {
        for (i = 0; i < 50; i++) {
            local_data[i] = (double)rand() / RAND_MAX;
        }
    }
    
    MPI_Bcast(local_data, 50, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("Process %d received broadcast array, first 3 elements: %.4f %.4f %.4f\n",
           rank, local_data[0], local_data[1], local_data[2]);
    
    // Final synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("MPI C test completed successfully!\n");
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
EOF
    
    local test_command="mpicc -o '$test_exe' '$test_file' && mpirun -np 8 '$test_exe'"
    run_test "$test_name" "$test_command" "Compile and run MPI C program with 8 processes"
}

# Test 11: MPI Performance and Scalability
test_mpi_performance() {
    local test_name="MPI_Performance"
    local test_file="${WORK_DIR}/test_mpi_performance.c"
    local test_exe="${WORK_DIR}/test_mpi_performance"
    
    # Create MPI performance test program
    cat > "$test_file" << 'EOF'
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size, i, j, k;
    int n = 1000;  // Matrix size
    double start_time, end_time, total_time;
    double *A, *B, *C, *local_A, *local_C;
    int local_n, local_start, local_end;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("MPI Performance Test Program\n");
        printf("============================\n");
        printf("Matrix size: %dx%d\n", n, n);
        printf("Number of processes: %d\n\n", size);
    }
    
    // Calculate local matrix size
    local_n = n / size;
    local_start = rank * local_n;
    local_end = (rank == size - 1) ? n : (rank + 1) * local_n;
    
    // Allocate memory
    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));
        
        // Initialize matrices
        for (i = 0; i < n * n; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
            C[i] = 0.0;
        }
    } else {
        A = NULL;
        B = (double*)malloc(n * n * sizeof(double));
        C = NULL;
        
        // Initialize B matrix
        for (i = 0; i < n * n; i++) {
            B[i] = (double)rand() / RAND_MAX;
        }
    }
    
    local_A = (double*)malloc(local_n * n * sizeof(double));
    local_C = (double*)malloc(local_n * n * sizeof(double));
    
    // Test 1: Matrix-Vector Multiplication
    if (rank == 0) {
        printf("Test 1: Matrix-Vector Multiplication\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Scatter matrix A
    MPI_Scatter(A, local_n * n, MPI_DOUBLE, local_A, local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast vector B (using first row as vector)
    MPI_Bcast(B, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform local matrix-vector multiplication
    for (i = 0; i < local_n; i++) {
        local_C[i] = 0.0;
        for (j = 0; j < n; j++) {
            local_C[i] += local_A[i * n + j] * B[j];
        }
    }
    
    // Gather results
    MPI_Gather(local_C, local_n, MPI_DOUBLE, C, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Matrix-vector multiplication completed in %.6f seconds\n", end_time - start_time);
    }
    
    // Test 2: All-to-All Communication
    if (rank == 0) {
        printf("\nTest 2: All-to-All Communication\n");
    }
    
    double *send_data = (double*)malloc(size * sizeof(double));
    double *recv_data = (double*)malloc(size * sizeof(double));
    
    for (i = 0; i < size; i++) {
        send_data[i] = (double)rank + i * 0.1;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Alltoall(send_data, 1, MPI_DOUBLE, recv_data, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("All-to-all communication completed in %.6f seconds\n", end_time - start_time);
    }
    
    // Test 3: Collective Reduction
    if (rank == 0) {
        printf("\nTest 3: Collective Reduction\n");
    }
    
    double local_sum = 0.0;
    for (i = 0; i < 1000000; i++) {
        local_sum += sin((double)i * 0.001);
    }
    
    double global_sum;
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Collective reduction completed in %.6f seconds\n", end_time - start_time);
        printf("Global sum: %.6f\n", global_sum);
    }
    
    // Test 4: Point-to-Point Bandwidth
    if (rank == 0) {
        printf("\nTest 4: Point-to-Point Bandwidth\n");
    }
    
    int message_size = 1000000;  // 1MB
    double *message = (double*)malloc(message_size * sizeof(double));
    
    for (i = 0; i < message_size; i++) {
        message[i] = (double)rand() / RAND_MAX;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    if (rank == 0 && size > 1) {
        MPI_Send(message, message_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(message, message_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0 && size > 1) {
        double bandwidth = (message_size * sizeof(double)) / (end_time - start_time) / (1024 * 1024);  // MB/s
        printf("Point-to-point bandwidth: %.2f MB/s\n", bandwidth);
    }
    
    // Test 5: Scalability Test
    if (rank == 0) {
        printf("\nTest 5: Scalability Test\n");
    }
    
    int iterations = 1000;
    double *local_data = (double*)malloc(1000 * sizeof(double));
    
    for (i = 0; i < 1000; i++) {
        local_data[i] = (double)rand() / RAND_MAX;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    for (k = 0; k < iterations; k++) {
        MPI_Allreduce(local_data, local_data, 1000, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Scalability test (%d iterations) completed in %.6f seconds\n", iterations, end_time - start_time);
        printf("Average time per iteration: %.6f seconds\n", (end_time - start_time) / iterations);
    }
    
    // Clean up
    free(local_A);
    free(local_C);
    free(B);
    if (rank == 0) {
        free(A);
        free(C);
    }
    free(send_data);
    free(recv_data);
    free(message);
    free(local_data);
    
    if (rank == 0) {
        printf("\nMPI Performance test completed successfully!\n");
    }
    
    MPI_Finalize();
    return 0;
}
EOF
    
    local test_command="mpicc -O2 -o '$test_exe' '$test_file' && mpirun -np 8 '$test_exe'"
    run_test "$test_name" "$test_command" "Compile and run MPI performance test with 8 processes"
}

# Check available software
check_software() {
    log_info "Checking available software..."
    
    echo "=== Software Availability ===" >> "$RESULTS_FILE"
    
    local software_list=("gfortran" "matlab" "python3" "Rscript" "stata" "mpif90" "mpirun" "mpicc")
    local software_names=("Fortran Compiler" "MATLAB" "Python 3" "R" "Stata" "MPI Fortran Compiler" "MPI Runtime" "MPI C Compiler")
    
    for i in "${!software_list[@]}"; do
        if command_exists "${software_list[$i]}"; then
            local version=""
            case "${software_list[$i]}" in
                "gfortran")
                    version=$("${software_list[$i]}" --version | head -n1 2>/dev/null || echo "Version unknown")
                    ;;
                "matlab")
                    version=$("${software_list[$i]}" -batch "version" 2>/dev/null | grep "MATLAB" | head -n1 || echo "Version unknown")
                    ;;
                "python3")
                    version=$("${software_list[$i]}" --version 2>/dev/null || echo "Version unknown")
                    ;;
                "Rscript")
                    version=$("${software_list[$i]}" --version 2>/dev/null | head -n1 || echo "Version unknown")
                    ;;
                "stata")
                    version=$("${software_list[$i]}" -b -e "version" 2>/dev/null | head -n1 || echo "Version unknown")
                    ;;
                "mpif90")
                    version=$("${software_list[$i]}" --version 2>/dev/null | head -n1 || echo "Version unknown")
                    ;;
                "mpirun")
                    version=$("${software_list[$i]}" --version 2>/dev/null | head -n1 || echo "Version unknown")
                    ;;
                "mpicc")
                    version=$("${software_list[$i]}" --version 2>/dev/null | head -n1 || echo "Version unknown")
                    ;;
            esac
            echo "${software_names[$i]}: Available - $version" >> "$RESULTS_FILE"
            log_success "${software_names[$i]} is available"
        else
            echo "${software_names[$i]}: Not available" >> "$RESULTS_FILE"
            log_warning "${software_names[$i]} is not available"
        fi
    done
    
    echo "" >> "$RESULTS_FILE"
}

# Generate final report
generate_report() {
    log_info "Generating final test report..."
    
    echo "=== Final Test Report ===" | tee -a "$LOG_FILE"
    echo "End Time: $(date)" | tee -a "$LOG_FILE"
    echo "Total Tests: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo "Passed: $PASSED_TESTS" | tee -a "$LOG_FILE"
    echo "Failed: $FAILED_TESTS" | tee -a "$LOG_FILE"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo "Overall Status: ALL TESTS PASSED" | tee -a "$LOG_FILE"
        echo -e "${GREEN}🎉 All tests completed successfully!${NC}" | tee -a "$LOG_FILE"
    else
        echo "Overall Status: SOME TESTS FAILED" | tee -a "$LOG_FILE"
        echo -e "${YELLOW}⚠️  $FAILED_TESTS test(s) failed. Check error details.${NC}" | tee -a "$LOG_FILE"
        
        echo "" | tee -a "$LOG_FILE"
        echo "Failed Tests:" | tee -a "$LOG_FILE"
        for test_name in "${!TEST_STATUS[@]}"; do
            if [ "${TEST_STATUS[$test_name]}" = "FAILED" ]; then
                echo "  - $test_name" | tee -a "$LOG_FILE"
            fi
        done
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "Detailed results available in: $RESULTS_FILE" | tee -a "$LOG_FILE"
    echo "Error details available in: $ERROR_LOG" | tee -a "$LOG_FILE"
    echo "Test log available in: $LOG_FILE" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    init_logging
    gather_system_info
    check_software
    
    # Run tests (continue on failure)
    log_info "Starting comprehensive test suite..."
    
    # Test Fortran
    if command_exists gfortran; then
        test_fortran
    else
        log_warning "Skipping Fortran test - gfortran not available"
    fi
    
    # Test MATLAB
    if command_exists matlab; then
        test_matlab
    else
        log_warning "Skipping MATLAB test - matlab not available"
    fi
    
    # Test Python
    if command_exists python3; then
        test_python
    else
        log_warning "Skipping Python test - python3 not available"
    fi
    
    # Test R
    if command_exists Rscript; then
        test_r
    else
        log_warning "Skipping R test - Rscript not available"
    fi
    
    # Test Stata
    if command_exists stata; then
        test_stata
    else
        log_warning "Skipping Stata test - stata not available"
    fi
    
    # MPI Tests Section
    log_info "Starting MPI parallel computing tests..."
    
    # Test MPI Fortran
    if command_exists mpif90 && command_exists mpirun; then
        test_mpi_fortran
    else
        log_warning "Skipping MPI Fortran test - mpif90 or mpirun not available"
    fi
    
    # Test MPI Python
    if command_exists mpirun && command_exists python3; then
        # Check if mpi4py is available
        if python3 -c "import mpi4py" 2>/dev/null; then
            test_mpi_python
        else
            log_warning "Skipping MPI Python test - mpi4py not available"
        fi
    else
        log_warning "Skipping MPI Python test - mpirun or python3 not available"
    fi
    
    # Test MPI R
    if command_exists mpirun && command_exists Rscript; then
        # Check if Rmpi is available
        if Rscript -e "library(Rmpi)" 2>/dev/null; then
            test_mpi_r
        else
            log_warning "Skipping MPI R test - Rmpi not available"
        fi
    else
        log_warning "Skipping MPI R test - mpirun or Rscript not available"
    fi
    
    # Test MPI MATLAB
    if command_exists matlab; then
        test_mpi_matlab
    else
        log_warning "Skipping MPI MATLAB test - matlab not available"
    fi
    
    # Test MPI C
    if command_exists mpicc && command_exists mpirun; then
        test_mpi_c
    else
        log_warning "Skipping MPI C test - mpicc or mpirun not available"
    fi
    
    # Test MPI Performance
    if command_exists mpicc && command_exists mpirun; then
        test_mpi_performance
    else
        log_warning "Skipping MPI Performance test - mpicc or mpirun not available"
    fi
    
    generate_report
}

# Execute main function
main "$@"
