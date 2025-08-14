#!/bin/bash
#SBATCH --job-name=BOINC_Apptainer
#SBATCH --output=%j_boinc_apptainer.out
#SBATCH --error=%j_boinc_apptainer.err
#SBATCH --export=ALL
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hoy@uchicago.edu

# Job description
echo "=========================================="
echo "BOINC Distributed Computing with Apptainer"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Job start time: $(date)"
echo "=========================================="

# Move to the submission directory
cd $SLURM_SUBMIT_DIR

# Create BOINC data directory
BOINC_DATA_DIR="$SLURM_SUBMIT_DIR/boinc_data_${SLURM_JOB_ID}"
mkdir -p $BOINC_DATA_DIR
echo "BOINC data directory: $BOINC_DATA_DIR"

# Set BOINC configuration
BOINC_PROJECT_URL="https://boinc.bakerlab.org/rosetta/"
BOINC_USERNAME="anonymous"
BOINC_PASSWORD="anonymous"

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "ERROR: Apptainer is not available on this system!"
    echo "Please ensure Apptainer is installed and accessible."
    exit 1
fi

echo "Apptainer version: $(apptainer --version)"

# Define container image path
CONTAINER_IMAGE="$SLURM_SUBMIT_DIR/boinc_client.sif"

# Pull the BOINC container image (if not already available)
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Pulling BOINC container image..."
    apptainer pull $CONTAINER_IMAGE docker://boinc/client:latest
else
    echo "Using existing container image: $CONTAINER_IMAGE"
fi

# Create a container configuration file
cat > $BOINC_DATA_DIR/boinc_config.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<cc_config>
    <options>
        <ncpus>$SLURM_CPUS_PER_TASK</ncpus>
        <max_ncpus_pct>100</max_ncpus_pct>
        <memory_limit>$(($SLURM_MEM_PER_NODE * 1024 * 1024))</memory_limit>
        <disk_interval>60</disk_interval>
        <disk_max_used_gb>10</disk_max_used_gb>
        <disk_min_free_gb>1</disk_min_free_gb>
        <vm_max_used_pct>75</vm_max_used_pct>
        <work_buf_min_days>0.1</work_buf_min_days>
        <work_buf_additional_days>0.5</work_buf_additional_days>
        <max_file_xfers>4</max_file_xfers>
        <max_file_xfers_per_project>2</max_file_xfers_per_project>
        <network_test_url>http://boinc.bakerlab.org/</network_test_url>
        <run_apps_manually>0</run_apps_manually>
        <run_gpu_apps>0</run_gpu_apps>
        <abort_jobs_on_exit>0</abort_jobs_on_exit>
        <exit_after_finish>0</exit_after_finish>
        <exit_before_start>0</exit_before_start>
        <exit_when_idle>0</exit_when_idle>
        <fetch_minimal_work>0</fetch_minimal_work>
        <http_1_0>0</http_1_0>
        <http_transfer_timeout>60</http_transfer_timeout>
        <http_transfer_timeout_bps>50000</http_transfer_timeout_bps>
        <max_event_log_lines>1000</max_event_log_lines>
        <max_file_xfers>4</max_file_xfers>
        <max_file_xfers_per_project>2</max_file_xfers_per_project>
        <max_stderr_file_size>1000000</max_stderr_file_size>
        <max_stdout_file_size>1000000</max_stdout_file_size>
        <net_end_hour>0</net_end_hour>
        <net_start_hour>0</net_start_hour>
        <network_test_url>http://boinc.bakerlab.org/</network_test_url>
        <os_random_only>0</os_random_only>
        <process_priority>0</process_priority>
        <process_priority_special>0</process_priority_special>
        <report_results_immediately>0</report_results_immediately>
        <run_apps_manually>0</run_apps_manually>
        <run_cpu_benchmarks>0</run_cpu_benchmarks>
        <run_gpu_apps>0</run_gpu_apps>
        <save_stats_days>30</save_stats_days>
        <show_advanced>0</show_advanced>
        <show_all_projects>0</show_all_projects>
        <show_workload>0</show_workload>
        <suppress_net_info>0</suppress_net_info>
        <suppress_schedule_rpc>0</suppress_schedule_rpc>
        <use_all_gpus>0</use_all_gpus>
        <use_certs>0</use_certs>
        <use_certs_only>0</use_certs_only>
        <use_http_1_1>1</use_http_1_1>
        <use_http_proxy>0</use_http_proxy>
        <use_https_proxy>0</use_https_proxy>
        <use_socks_proxy>0</use_socks_proxy>
        <use_web_proxy>0</use_web_proxy>
        <verbosity>99</verbosity>
    </options>
</cc_config>
EOF

echo "BOINC configuration created"

# Create a script to run inside the container
cat > $BOINC_DATA_DIR/run_boinc.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting BOINC client..."

# Initialize BOINC client
boinc --daemon --dir /var/lib/boinc --allow_remote_gui_rpc

# Wait a moment for the daemon to start
sleep 5

# Attach to the project
echo "Attaching to project: $BOINC_PROJECT_URL"
boinccmd --project_attach $BOINC_PROJECT_URL $BOINC_USERNAME $BOINC_PASSWORD

# Wait for attachment to complete
sleep 10

# Check project status
echo "Project status:"
boinccmd --get_project_status

# Start processing work units
echo "Starting work unit processing..."
boinccmd --set_run_mode always

# Monitor progress
while true; do
    echo "=== BOINC Status Report ==="
    echo "Time: $(date)"
    
    # Get project status
    boinccmd --get_project_status
    
    # Get task list
    echo "Active tasks:"
    boinccmd --get_tasks
    
    # Get disk usage
    echo "Disk usage:"
    df -h /var/lib/boinc
    
    # Sleep for 5 minutes before next check
    sleep 300
done
EOF

chmod +x $BOINC_DATA_DIR/run_boinc.sh

# Run the BOINC container with Apptainer
echo "Starting BOINC container with Apptainer..."
apptainer exec \
    --bind $BOINC_DATA_DIR:/var/lib/boinc \
    --bind $BOINC_DATA_DIR/boinc_config.xml:/etc/boinc-client/cc_config.xml \
    --env BOINC_PROJECT_URL=$BOINC_PROJECT_URL \
    --env BOINC_USERNAME=$BOINC_USERNAME \
    --env BOINC_PASSWORD=$BOINC_PASSWORD \
    --env BOINC_DATA_DIR=/var/lib/boinc \
    --pwd /var/lib/boinc \
    $CONTAINER_IMAGE \
    /var/lib/boinc/run_boinc.sh

# Container exit handling
CONTAINER_EXIT_CODE=$?

echo "=========================================="
echo "BOINC job completed!"
echo "Job end time: $(date)"
echo "Container exit code: $CONTAINER_EXIT_CODE"
echo "Data directory: $BOINC_DATA_DIR"
echo "=========================================="

# Create a summary report
{
    echo "BOINC Job Summary (Apptainer)"
    echo "============================="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Start time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo 'Unknown')"
    echo "End time: $(date)"
    echo "Node: $SLURM_NODELIST"
    echo "CPUs used: $SLURM_CPUS_PER_TASK"
    echo "Memory used: $SLURM_MEM_PER_NODE MB"
    echo "Container exit code: $CONTAINER_EXIT_CODE"
    echo "BOINC data directory: $BOINC_DATA_DIR"
    echo "Container image: $CONTAINER_IMAGE"
    echo ""
    echo "Files generated:"
    ls -la $BOINC_DATA_DIR/ 2>/dev/null || echo "No files found"
} > boinc_apptainer_summary_${SLURM_JOB_ID}.txt

echo "Summary saved to: boinc_apptainer_summary_${SLURM_JOB_ID}.txt"

# Cleanup (optional - uncomment if you want to remove data after job completion)
# echo "Cleaning up BOINC data directory..."
# rm -rf $BOINC_DATA_DIR

exit $CONTAINER_EXIT_CODE
