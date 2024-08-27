import os
import requests
import pandas as pd
import random
from opencage.geocoder import OpenCageGeocode
import io
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

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

def get_regular_coordinates():
    # Charger les données de la carte du monde
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Définir les limites de la grille
    min_lon, min_lat, max_lon, max_lat = -180, -90, 180, 90
    grid_spacing = 10  # espacement de 10 degrés

    # Générer les points de la grille
    lons = np.arange(min_lon, max_lon, grid_spacing)
    lats = np.arange(min_lat, max_lat, grid_spacing)
    grid_points = [Point(lon, lat) for lon in lons for lat in lats]

    # Créer un GeoDataFrame pour les points de la grille
    grid_df = gpd.GeoDataFrame(geometry=grid_points)

    # Extraire les centroïdes des pays
    centroids = world.centroid

    # Créer un GeoDataFrame pour les centroïdes
    centroids_df = gpd.GeoDataFrame(geometry=centroids)

    # Fusionner les points de la grille et les centroïdes
    all_points = pd.concat([grid_df, centroids_df], ignore_index=True)

    # Vérifier quels points sont dans les frontières des pays
    points_in_world = gpd.sjoin(all_points, world, op='within')
    return points_in_world

def coordinates_in_country(country_name, points_in_world):
    coordinates = []
    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    df_points_in_country = points_in_world[points_in_world['iso_a3']==iso_code ]

    if len(df_points_in_country)!=0:
        for point in df_points_in_country['geometry']:
            coordinates.append((point.y, point.x))
    return coordinates

def coordinates_in_state(country_name, state_name, points_in_world):
    coordinates = []
    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    df_points_in_country = points_in_world[points_in_world['iso_a3']==iso_code ]
    df_points_in_state = df_points_in_country[df_points_in_country['state_code']==state_name]
    if len(df_points_in_state)!=0:
        for point in df_points_in_state['geometry']:
            coordinates.append((point.y, point.x))
    return coordinates


def grid_coordinates(spacing, plot = False, savefile = False):
    # Charger les données de la carte du monde
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world.to_crs(epsg=4326)
    # Définir les limites de la grille
    min_lon, min_lat, max_lon, max_lat = -180, -90, 180, 90
    grid_spacing = spacing  # espacement de 10 degrés

    # Générer les points de la grille
    lons = np.arange(min_lon, max_lon, grid_spacing)
    lats = np.arange(min_lat, max_lat, grid_spacing)
    grid_points = [Point(lon, lat) for lon in lons for lat in lats]

    # Créer un GeoDataFrame pour les points de la grille
    grid_df = gpd.GeoDataFrame(geometry=grid_points, crs=world.crs)

    # Extraire les centroïdes des pays
    centroids = world.centroid

    # Créer un GeoDataFrame pour les centroïdes
    centroids_df = gpd.GeoDataFrame(geometry=centroids, crs=world.crs)

    # Ajouter le centroïde de la France métropolitaine
    france = world[world.name == "France"]
    # Filtrer la France métropolitaine (la plus grande géométrie)
    france_metropolitan = france.loc[france.geometry.area.idxmax()]
    france_centroid = france_metropolitan.geometry.centroid

    # Créer un GeoDataFrame pour le centroïde de la France métropolitaine
    france_centroid_df = gpd.GeoDataFrame(geometry=[france_centroid], crs=world.crs)

    # Fusionner les points de la grille et les centroïdes
    all_points = pd.concat([grid_df, centroids_df, france_centroid_df], ignore_index=True)

    # Filtrer les points pour enlever ceux situés en Antarctique
    all_points['latitude'] = all_points.geometry.apply(lambda p: p.y)
    all_points = all_points[all_points['latitude'] > -60]

    # Vérifier quels points sont dans les frontières des pays
    points_in_world = gpd.sjoin(all_points, world, op='within', how='inner')

    if savefile:
        # Sauvegarder la grille pour future utilisation
        output_path = 'grid_with_centroids.geojson'
        points_in_world.to_file(output_path, driver='GeoJSON')
    if plot:
        # Visualiser la grille et les centroïdes sur la carte du monde
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='lightgray')
        points_in_world.plot(ax=ax, color='red', markersize=5)

        plt.title('Regular Grid of Coordinates with Country Centroids')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    return points_in_world

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
    

def fetch_and_average_data_ren_ninja(country, num_locations, technologies, points_in_world, state = None, year=2020, save = False, coordinates = 'grid'):
    if  coordinates == 'random' :
        coordinates_list = get_random_coordinates(country, num_locations)
    elif coordinates =='grid' : 
        if state :
            coordinates_list = coordinates_in_state(country,state, points_in_world)
        else:
            coordinates_list = coordinates_in_country(country, points_in_world)
        num_locations = len(coordinates_list)
    else:
        raise ValueError('Unknown input for coordinates.')
    all_data = {}

    for tech in technologies:
        tech_data = []
        for lat, lon in coordinates_list:
            try:
                data = get_renewable_data(lat, lon, tech, year)
                tech_data.append(data)
            except Exception as e:
                print(f"Error fetching data for location ({lat}, {lon}) with technology {tech}: {e}")
        
        if tech_data:
            combined_df = pd.concat(tech_data).groupby('time').mean().reset_index()
            all_data[tech] = combined_df

        if save : 
            
            if not os.path.exists(f'../input_time_series/{country}'):
                os.makedirs(f'../input_time_series/{country}')
            for tech, df in all_data.items():
                if state:
                    file_save = f'../input_time_series/{country}/ren_ninja_{num_locations}_{coordinates}_locations_averaged_{tech}_{country}_{state}_{year}.xlsx'
                else :
                    file_save = f'../input_time_series/{country}/ren_ninja_{num_locations}_{coordinates}_locations_averaged_{tech}_{country}_{year}.xlsx'
                df['electricity'].to_excel(file_save, index=False)

    return 

