# Weather Data Collection for SLURM Cluster

This example demonstrates the collection of a small sample of data from NOAA Weather.gov API for the 20 most-populated US cities using a distributed SLURM cluster approach. Each node processes one city to verify and document that data collection via https is functional.

## Features

- **Distributed Processing**: Uses 20 SLURM nodes, each processing one city
- **Real-time Weather Data**: Collects current weather conditions from NOAA Weather.gov API
- **Comprehensive Data**: Includes temperature, humidity, wind speed, pressure, and more
- **Logging**: Detailed logging for each node with separate log files
- **Data Persistence**: Saves weather data as JSON files with timestamps
- **Error Handling**: Robust error handling for API failures and network issues

## Cities Covered

The system processes the 20 most-populated US cities:
1. New York, NY
2. Los Angeles, CA
3. Chicago, IL
4. Houston, TX
5. Phoenix, AZ
6. Philadelphia, PA
7. San Antonio, TX
8. San Diego, CA
9. Dallas, TX
10. San Jose, CA
11. Austin, TX
12. Jacksonville, FL
13. Fort Worth, TX
14. Columbus, OH
15. Charlotte, NC
16. San Francisco, CA
17. Indianapolis, IN
18. Seattle, WA
19. Denver, CO
20. Washington, DC

## Prerequisites

1. **SLURM Cluster**: Access to a SLURM cluster with CPU partition
2. **Python Environment**: Use a conda environment
3. **Internet Access**: Cluster nodes need access to api.weather.gov

## Setup

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

## Usage

### Submit the Job

```bash
sbatch weather.slurm
```

### Monitor the Job

```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f weather_<job_id>.out

# Check job logs
tail -f weather_<job_id>.err
```

### Check Results

After the job completes, you'll find:

- **Weather data files**: `weather_data_node_X_YYYYMMDD_HHMMSS.json` for each city
- **Log files**: `weather_node_X.log` for each node
- **Summary file**: `weather_summary_<job_id>.txt` with overall job summary
- **Output directory**: `weather_data_<job_id>_YYYYMMDD_HHMMSS/` containing all results

## Output Format

Each weather data JSON file contains:

```json
{
  "city": "New York, NY",
  "timestamp": "2024-01-15T14:30:00.123456",
  "temperature": 45,
  "temperature_unit": "F",
  "short_forecast": "Mostly sunny",
  "detailed_forecast": "Mostly sunny, with a high near 45. North wind around 8 mph.",
  "wind_speed": "8 mph",
  "wind_direction": "N",
  "start_time": "2024-01-15T09:00:00-05:00",
  "end_time": "2024-01-15T18:00:00-05:00",
  "is_daytime": true
}
```

## SLURM Configuration

The job is configured with:
- **20 nodes** in the CPU partition
- **1 task per node** (each node processes one city)
- **2GB memory** per node
- **30-minute time limit**
- **Email notifications** for job status

## Customization

### Modify Cities

Edit the `CITIES` list in `weather.py` to change which cities are processed.

### Adjust Resources

Modify the SLURM parameters in `weather.slurm`:
- `--nodes`: Number of nodes (must match number of cities)
- `--mem`: Memory per node
- `--time`: Job time limit
- `--partition`: SLURM partition to use

### Change API Parameters

Modify the `get_weather_data()` function in `weather.py` to:
- Add additional weather parameters
- Modify API endpoint
- Change forecast period (currently uses first period)

## Troubleshooting

### Common Issues

1. **Network Connectivity**
   ```
   Error fetching weather data: Connection timeout
   ```
   Solution: Internet connectivity might be a problem!  Report the issue to ssc_server_support@listhost.uchicago.edu

2. **API Rate Limits**
   ```
   Error fetching weather data: 429 Too Many Requests
   ```
   Solution: Weather.gov has rate limits. You should not encounter those rate limits unless you have altered the scripts dramatically.

3. **No Forecast Data**
   ```
   No forecast periods found for [City Name]
   ```
   Solution: Don't replace the city names provided with cities that are not on the map.

4. **Insufficient Nodes**
   ```
   Not enough nodes available for the job.  It sits in the queue forever.
   ```
   Solution: Adjust the number of cities and the number of nodes so that it will run with available resources.

### Debug Mode

To run in debug mode, modify the logging level in `weather.py`:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Performance

- **Typical runtime**: 1-5 minutes depending on API response times
- **Data size**: ~1KB per city (JSON format)
- **Network usage**: ~20 API calls (one per city)
- **CPU usage**: Minimal (mostly I/O bound)

## License

This project is part of the sbatch_examples repository of the Social Sciences Computing Services team at the University of Chicago.
