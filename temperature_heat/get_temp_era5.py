import cdsapi
import geopandas as gpd
import os
import pandas as pd
# import statsmodels.api as sm

def download_era5_temperature_data(country_bounding_box, country_name, start_year, end_year):
    file_name = f'era5_temperature_{country_name}_{start_year}_{end_year}.nc'
    if os.path.exists(file_name):
        print('File already exists')
        return
    else: 
        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': '2m_temperature',
                'year': [str(year) for year in range(start_year, end_year + 1)],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00', '03:00',
                    '04:00', '05:00', '06:00', '07:00',
                    '08:00', '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00', '15:00',
                    '16:00', '17:00', '18:00', '19:00',
                    '20:00', '21:00', '22:00', '23:00',
                ],
                'format': 'netcdf',
                'area': country_bounding_box,  # Latitude/longitude bounding box for the selected country
            },
            f'era5_temperature_{country_name}_{start_year}_{end_year}.nc')


def degree_days(temperature_ts, t_heat, t_cool):

    df = pd.Dataframe(temperature_ts)

    # Calculer les heating degree days (HDD) et les cooling degree days (CDD) pour chaque jour
    df['HDD'] = df['temperature'].apply(lambda x: max(t_heat - x, 0))
    df['CDD'] = df['temperature'].apply(lambda x: max(x - t_cool, 0))

    # Calculer les heating degree days (HDD) et les cooling degree days (CDD) cumulés au fil du temps
    df['cumulative_HDD'] = df['HDD'].cumsum()
    df['cumulative_CDD'] = df['CDD'].cumsum()

    return df


# def fit_reg_morceaux(df, t_heat, t_cool):
#     # Diviser les données en deux ensembles : en dessous de 18°C et au-dessus de 22°C
#     df_below_18 = df[df['temperature'] < 18]
#     df_above_22 = df[df['temperature'] > 22]

#     # Ajouter une constante à la matrice X pour l'estimation du modèle
#     X_below_18 = sm.add_constant(df_below_18['temperature'])
#     X_above_22 = sm.add_constant(df_above_22['temperature'])

#     # Ajuster un modèle de régression linéaire pour les températures en dessous de 18°C
#     model_below_18 = sm.OLS(df_below_18['electricity_consumption'], X_below_18).fit()

#     # Ajuster un modèle de régression linéaire pour les températures au-dessus de 22°C
#     model_above_22 = sm.OLS(df_above_22['electricity_consumption'], X_above_22).fit()
#     # Afficher les résultats des modèles
#     print("Modèle pour les températures en dessous de 18°C :")
#     print(model_below_18.summary())

#     print("\nModèle pour les températures au-dessus de 22°C :")
#     print(model_above_22.summary())



