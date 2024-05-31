import os
import requests
import pandas as pd
import random
from opencage.geocoder import OpenCageGeocode
import io

# Your Renewable Ninja API token
api_token = 'e2be88b4f8ae85f401fdf6b205165098e3cf4e37'

opencage_api_key ='b42473080f3a4bdb9033973e58fea766'
# Define the endpoint
base_url = ' https://www.renewables.ninja/api/data/'

# Function to get random coordinates within a country
def get_random_coordinates(country, num_locations):
    geocoder = OpenCageGeocode(opencage_api_key)
    result = geocoder.geocode(country, no_annotations='1')
    
    if not result:
        raise ValueError(f"Cannot find location for the country: {country}")
    
    first_result = result[0]
    # print(first_result)
    # if 'geometry' not in first_result or 'bounds' not in first_result['geometry']:
    #     raise ValueError(f"Cannot find bounding box information for {country}")
    
    bbox = first_result['bounds']
    min_lat, max_lat = bbox['southwest']['lat'], bbox['northeast']['lat']
    min_lon, max_lon = bbox['southwest']['lng'], bbox['northeast']['lng']
    
    coordinates = []
    for _ in range(num_locations):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        coordinates.append((lat, lon))
    return coordinates

def get_renewable_data(lat, lon, technology, year=2021):
    headers = {
        'Authorization': 'Token ' + api_token,
    }
    params = {
        'lat': lat,
        'lon': lon,
        'date_from': f'{year}-01-01',
        'date_to': f'{year}-12-31',
        'dataset': 'merra2',
        'capacity': 1,
        'format': 'csv'
    }
    
    if technology == 'pv':
        params.update({
            'system_loss': 0.1,
            'tracking': 0,
            'tilt': 35,
            'azim': 180
        })
    elif technology == 'wind':
        params.update({
            'height': 80,  # Specify an appropriate height
            'turbine': 'Vestas V90 2000'  # Specify a turbine model
        })
    
    url = f"{base_url}{technology}"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        # Skip potential metadata and read the CSV data correctly
        csv_data = response.text.split('\n', 1)[1]
        return pd.read_csv(io.StringIO(csv_data), skiprows=2, index_col=0)
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
    

def fetch_and_average_data_ren_ninja(country, num_locations, technologies, year=2020, save = False):
    coordinates = get_random_coordinates(country, num_locations)
    all_data = {}

    for tech in technologies:
        tech_data = []
        for lat, lon in coordinates:
            try:
                data = get_renewable_data(lat, lon, tech, year)
                tech_data.append(data)
            except Exception as e:
                print(f"Error fetching data for location ({lat}, {lon}) with technology {tech}: {e}")
        
        if tech_data:
            combined_df = pd.concat(tech_data).groupby('time').mean().reset_index()
            all_data[tech] = combined_df

        if save : 
            if os.path.exists(f'../input_time_series/{country}'):
                pass
            else :
                os.makedirs(f'../input_time_series/{country}')
            for tech, df in all_data.items():
                df['electricity'].to_excel(f'../input_time_series/{country}/ren_ninja_{num_locations}_locations_averaged_{tech}_{country}_{year}.xlsx', index=False)

    return 

