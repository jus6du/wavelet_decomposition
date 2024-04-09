import pandas as pd

def get_heating_coeff_ts(country_code, year):
    if year<=2016:
        file_path = '../../../DATA/OSMOSE_Dataset/OSMOSE_DATASET/load/heating_coeff/'
        file_name = f'heating_coeff_{year}.csv'
        data = pd.read_csv(file_path+file_name)
        data_country = data[data['country']==country_code].set_index('timestamp')['heating_coeff']
        return data_country
    else : 
        print(f'The year {year} is not included in the dataset')
        return