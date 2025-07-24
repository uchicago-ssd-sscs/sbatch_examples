#!/usr/bin/env Rscript

# R CPU Utilization Test Script
# This script performs various computationally intensive tasks using only base R
# No additional packages required

cat("Starting R CPU utilization test...\n")
start_time <- Sys.time()

# Test 1: Large matrix operations
cat("Test 1: Large matrix operations...\n")
n <- 2000
A <- matrix(rnorm(n^2), n, n)
B <- matrix(rnorm(n^2), n, n)

# Matrix multiplication
C <- A %*% B
cat("Matrix multiplication completed. Result dimensions:", dim(C), "\n")

# Matrix inversion
D <- solve(A)
cat("Matrix inversion completed.\n")

# Eigenvalue decomposition
eigen_result <- eigen(A)
cat("Eigenvalue decomposition completed. Largest eigenvalue magnitude:", max(Mod(eigen_result$values)), "\n")

# Test 2: Statistical computations
cat("Test 2: Statistical computations...\n")
x <- rnorm(1000000)
y <- rnorm(1000000)

# Correlation
cor_xy <- cor(x, y)
cat("Correlation coefficient:", cor_xy, "\n")

# Linear regression
lm_result <- lm(y ~ x)
cat("Linear regression completed. R-squared:", summary(lm_result)$r.squared, "\n")

# Test 3: Monte Carlo simulation
cat("Test 3: Monte Carlo simulation...\n")
n_sim <- 1000000
pi_estimate <- 0
inside_circle <- 0

for (i in 1:n_sim) {
    x <- runif(1, -1, 1)
    y <- runif(1, -1, 1)
    if (x^2 + y^2 <= 1) {
        inside_circle <- inside_circle + 1
    }
}

pi_estimate <- 4 * inside_circle / n_sim
cat("Monte Carlo Pi estimation:", pi_estimate, "(actual pi ≈ 3.14159)\n")

# Test 4: Sorting and searching
cat("Test 4: Sorting and searching...\n")
large_vector <- rnorm(500000)
sorted_vector <- sort(large_vector)
cat("Vector sorted. Min:", min(sorted_vector), "Max:", max(sorted_vector), "\n")

# Binary search simulation
search_value <- 0.5
binary_search_result <- which.min(abs(sorted_vector - search_value))
cat("Closest value to", search_value, "found at position", binary_search_result, "\n")

# Test 5: String operations
cat("Test 5: String operations...\n")
words <- paste0("word_", 1:100000)
cat("Generated", length(words), "words\n")

# String concatenation
long_string <- paste(words, collapse = " ")
cat("String concatenation completed. Length:", nchar(long_string), "\n")

# Test 6: Recursive function (Fibonacci)
cat("Test 6: Recursive computations...\n")
fibonacci <- function(n) {
    if (n <= 1) return(n)
    return(fibonacci(n-1) + fibonacci(n-2))
}

# Calculate Fibonacci numbers (this will be computationally intensive)
fib_results <- numeric(35)
for (i in 1:35) {
    fib_results[i] <- fibonacci(i)
}
cat("Fibonacci sequence calculated up to F(35):", fib_results[35], "\n")

# Test 7: Prime number generation
cat("Test 7: Prime number generation...\n")
is_prime <- function(n) {
    if (n < 2) return(FALSE)
    if (n == 2) return(TRUE)
    if (n %% 2 == 0) return(FALSE)
    for (i in 3:sqrt(n)) {
        if (n %% i == 0) return(FALSE)
    }
    return(TRUE)
}

primes <- numeric(0)
n <- 2
count <- 0
while (count < 1000) {
    if (is_prime(n)) {
        count <- count + 1
        primes[count] <- n
    }
    n <- n + 1
}
cat("Generated", length(primes), "prime numbers. Largest:", max(primes), "\n")

# Summary
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "secs")
cat("\n=== CPU Test Summary ===\n")
cat("Total execution time:", round(duration, 2), "seconds\n")
cat("CPU utilization test completed successfully!\n")

# Print system information
cat("\n=== System Information ===\n")
cat("R version:", R.version.string, "\n")
cat("Platform:", R.version$platform, "\n")
cat("Number of cores detected:", parallel::detectCores(), "\n")
cat("Memory usage:", format(object.size(ls()), units = "MB"), "\n") 