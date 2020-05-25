import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
This function :
    - Interpolate on greater number of points
    - Normalize each time series for each year
    - stact all years in a dictionnay : stacked data
For example  Nyears PV data are accessible in stacked_data[Solaire]
'''
def import_excel(path_input_data,input_file, dpd ,ndpd, dpy, time_series, interp=True):
    '''
    interp : Shall we interpolate drom dpd to ndpd. True or False
    dpd : data per day
    ndpd : new sata per day
    time_series : Which time series we want to import ? Consommasion, Eolien, Solaire

    Returns a disctionnary with stacked time series over the N years of the excel file. Each year is normalized and interpolated (if true)
    '''
    input_data = pd.ExcelFile(path_input_data + input_file)
    years = input_data.sheet_names
    NbYears = len(years)


    print('There is '+ str(NbYears) + ' years imported') ; print(years)

    stacked_data = {}
    for energy_ts in time_series:
        tmp_data = []
        for yr in years: # loop over the sheets of the excel file
            df = pd.read_excel(path_input_data + input_file,
                                sheet_name = yr,
                                name = True,
                               skiprows = 1) # skip firs row

            # Cut leap year
            one_yr = df[energy_ts].values
            one_yr = one_yr[0:dpd*dpy]

            # Normalize
            one_yr = one_yr/np.mean(one_yr) # Normalize yearly data to 1 MW
            # Remove the mean
            # one_yr = one_yr - np.mean(one_yr)

            #interpolate number of data per year. Interpolate one year after the other
            if interp:
                oldx = np.arange(0, dpy, 1./dpd)
                newx = np.arange(0, dpy, 1./ndpd)
                newy = np.interp(newx, oldx, one_yr)
                newy = np.interp(newx, oldx, one_yr)
            else:
                newy = one_d

            tmp_data = tmp_data +  newy.tolist()
        stacked_data[energy_ts] = np.array(tmp_data)

    # make sure that all signals have the same length
    lengths = [len(v) for v in stacked_data.values()]
    assert all(x==lengths[0] for x in lengths), 'all signals dont have the same length'

    lengths = [len(v) for v in stacked_data.values()]

    sns.set()
    time = np.arange(0, NbYears, 1./(ndpd*dpy))
    for energy_ts in time_series:
        plt.figure()
        plt.plot(time, stacked_data[energy_ts])
        plt.title(energy_ts)
        plt.xlabel('time (year)')
        plt.ylabel('Power normalized to 1 MW')
        plt.show()

    return stacked_data, years
