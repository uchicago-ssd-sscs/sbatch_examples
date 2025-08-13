#!/bin/bash
#SBATCH --job-name=WeatherDataCollection
#SBATCH --output=%j_weather.out
#SBATCH --error=%j_weather.err
#SBATCH --export=ALL
#SBATCH --partition=CPU
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hoy@uchicago.edu

# Job description
echo "=========================================="
echo "Weather Data Collection Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Job start time: $(date)"
echo "=========================================="

# Move to the submission directory
cd $SLURM_SUBMIT_DIR

# Load Python environment (adjust based on your cluster's available modules)
# module load python/anaconda3
conda init
conda activate cpu

# Note: No API key required for NOAA Weather.gov API
echo "Using NOAA Weather.gov API (no API key required)"

# Create output directory for this job
OUTPUT_DIR="weather_data_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo "Starting weather data collection across $SLURM_NTASKS nodes..."

# Run the weather collection script on all nodes
# Each node will process one city based on SLURM_PROCID
srun python ../weather.py

# Wait for all tasks to complete
wait

echo "=========================================="
echo "Weather data collection completed!"
echo "Job end time: $(date)"
echo "Output files saved in: $OUTPUT_DIR"
echo "=========================================="

# List the output files
echo "Generated files:"
ls -la *.json *.log 2>/dev/null || echo "No output files found"

# Optional: Create a summary file
echo "Creating summary..."
{
    echo "Weather Data Collection Summary"
    echo "================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Start time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo 'Unknown')"
    echo "End time: $(date)"
    echo "Nodes used: $SLURM_NNODES"
    echo "Tasks completed: $SLURM_NTASKS"
    echo ""
    echo "Cities processed:"
    for i in $(seq 0 $((SLURM_NTASKS-1))); do
        if [ -f "weather_data_node_${i}_"*.json ]; then
            echo "  Node $i: $(ls weather_data_node_${i}_*.json | head -1)"
        else
            echo "  Node $i: No data file found"
        fi
    done
} > weather_summary_${SLURM_JOB_ID}.txt

echo "Summary saved to: weather_summary_${SLURM_JOB_ID}.txt"
