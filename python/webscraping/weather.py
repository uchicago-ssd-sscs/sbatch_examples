#!/usr/bin/env python3
"""
Weather Data Collection Script for SLURM Cluster
Collects weather data from NOAA Weather.gov API for the 20 most-populated US cities.
Each SLURM node processes one city based on SLURM_PROCID.
"""

import os
import sys
import json
import requests
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'weather_node_{os.environ.get("SLURM_PROCID", "unknown")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# The 20 most-populated US cities with their coordinates
CITIES = [
    {"name": "New York", "state": "NY", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "state": "CA", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "state": "IL", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "state": "TX", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "state": "AZ", "lat": 33.4484, "lon": -112.0740},
    {"name": "Philadelphia", "state": "PA", "lat": 39.9526, "lon": -75.1652},
    {"name": "San Antonio", "state": "TX", "lat": 29.4241, "lon": -98.4936},
    {"name": "San Diego", "state": "CA", "lat": 32.7157, "lon": -117.1611},
    {"name": "Dallas", "state": "TX", "lat": 32.7767, "lon": -96.7970},
    {"name": "San Jose", "state": "CA", "lat": 37.3382, "lon": -121.8863},
    {"name": "Austin", "state": "TX", "lat": 30.2672, "lon": -97.7431},
    {"name": "Jacksonville", "state": "FL", "lat": 30.3322, "lon": -81.6557},
    {"name": "Fort Worth", "state": "TX", "lat": 32.7555, "lon": -97.3308},
    {"name": "Columbus", "state": "OH", "lat": 39.9612, "lon": -82.9988},
    {"name": "Charlotte", "state": "NC", "lat": 35.2271, "lon": -80.8431},
    {"name": "San Francisco", "state": "CA", "lat": 37.7749, "lon": -122.4194},
    {"name": "Indianapolis", "state": "IN", "lat": 39.7684, "lon": -86.1581},
    {"name": "Seattle", "state": "WA", "lat": 47.6062, "lon": -122.3321},
    {"name": "Denver", "state": "CO", "lat": 39.7392, "lon": -104.9903},
    {"name": "Washington", "state": "DC", "lat": 38.9072, "lon": -77.0369}
]

def get_weather_data(city):
    """
    Fetch weather data from NOAA Weather.gov API for a given city.
    
    Args:
        city (dict): City information with lat/lon coordinates
    
    Returns:
        dict: Weather data or None if error
    """
    # Weather.gov API endpoint for points
    points_url = f"https://api.weather.gov/points/{city['lat']},{city['lon']}"
    
    try:
        logger.info(f"Fetching weather data for {city['name']}, {city['state']}")
        
        # First, get the forecast URL for this location
        logger.info(f"Getting forecast URL from: {points_url}")
        response = requests.get(points_url, timeout=30)
        response.raise_for_status()
        
        points_data = response.json()
        forecast_url = points_data['properties']['forecast']
        
        # Get the current weather data
        logger.info(f"Getting forecast from: {forecast_url}")
        forecast_response = requests.get(forecast_url, timeout=30)
        forecast_response.raise_for_status()
        
        forecast_data = forecast_response.json()
        
        # Extract current conditions (first period in the forecast)
        if 'properties' in forecast_data and 'periods' in forecast_data['properties']:
            current_period = forecast_data['properties']['periods'][0]
            
            # Extract relevant weather information
            weather_info = {
                'city': f"{city['name']}, {city['state']}",
                'timestamp': datetime.now().isoformat(),
                'temperature': current_period.get('temperature'),
                'temperature_unit': current_period.get('temperatureUnit', 'F'),
                'short_forecast': current_period.get('shortForecast', 'N/A'),
                'detailed_forecast': current_period.get('detailedForecast', 'N/A'),
                'wind_speed': current_period.get('windSpeed', 'N/A'),
                'wind_direction': current_period.get('windDirection', 'N/A'),
                'start_time': current_period.get('startTime', 'N/A'),
                'end_time': current_period.get('endTime', 'N/A'),
                'is_daytime': current_period.get('isDaytime', 'N/A')
            }
            
            logger.info(f"Successfully retrieved weather data for {city['name']}")
            return weather_info
        else:
            logger.error(f"No forecast periods found for {city['name']}")
            return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data for {city['name']}: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing weather data for {city['name']}: {e}")
        return None

def save_weather_data(weather_data, node_id):
    """
    Save weather data to a JSON file.
    
    Args:
        weather_data (dict): Weather data to save
        node_id (int): SLURM node ID
    """
    if weather_data is None:
        return
    
    filename = f"weather_data_node_{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        logger.info(f"Weather data saved to {filename}")
    except IOError as e:
        logger.error(f"Error saving weather data: {e}")

def main():
    """Main function to collect weather data for assigned city."""
    
    # Get SLURM process ID to determine which city to process
    slurm_procid = int(os.environ.get('SLURM_PROCID', 0))
    slurm_ntasks = int(os.environ.get('SLURM_NTASKS', 20))
    
    logger.info(f"Starting weather data collection on node {slurm_procid}")
    logger.info(f"Total nodes: {slurm_ntasks}")
    
    # Validate that we have enough cities for all nodes
    if slurm_ntasks > len(CITIES):
        logger.error(f"Not enough cities ({len(CITIES)}) for {slurm_ntasks} nodes")
        sys.exit(1)
    
    # Get the city assigned to this node
    if slurm_procid >= len(CITIES):
        logger.error(f"Node {slurm_procid} exceeds number of cities ({len(CITIES)})")
        sys.exit(1)
    
    city = CITIES[slurm_procid]
    logger.info(f"Node {slurm_procid} assigned to {city['name']}, {city['state']}")
    
    # Collect weather data (no API key needed for Weather.gov)
    weather_data = get_weather_data(city)
    
    # Save the data
    save_weather_data(weather_data, slurm_procid)
    
    # Print summary
    if weather_data:
        print(f"\n=== Weather Summary for {weather_data['city']} ===")
        print(f"Temperature: {weather_data['temperature']}°{weather_data['temperature_unit']}")
        print(f"Forecast: {weather_data['short_forecast']}")
        print(f"Wind: {weather_data['wind_speed']} {weather_data['wind_direction']}")
        print(f"Time Period: {weather_data['start_time']} to {weather_data['end_time']}")
        print(f"Daytime: {weather_data['is_daytime']}")
        print(f"Timestamp: {weather_data['timestamp']}")
    else:
        print(f"Failed to collect weather data for {city['name']}, {city['state']}")
    
    logger.info(f"Weather data collection completed for node {slurm_procid}")

if __name__ == "__main__":
    main()
