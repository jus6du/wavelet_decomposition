import os
import pandas as pd
import openpyxl as xl

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def stack_time_series(excel_file):
    wb = xl.load_workbook(excel_file)
    df=pd.DataFrame()
    for sheet in wb.worksheets:
        df_to_add=pd.read_excel(excel_file,shett_name=sheet.title)
        df = pd.concat([df,df_to_add], axis=1)
    return df

def fill_missing_values(start_date, end_date, data, method = 'linear'):
    date_range = pd.date_range(start=pd.to_datetime("start_date"), end=pd.to_datetime("end_date"), freq="h")
    missing_dates = date_range[~date_range.isin(data.index)]
    print('There are ' + str(len(missing_dates))+' missing values in the time series.')
    data_reindexed = data.reindex(date_range)
    data_interpolated = data_reindexed.interpolate(method="linear")
    return data_interpolated

