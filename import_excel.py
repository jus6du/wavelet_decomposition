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
def import_excel(path_input_data,input_file, dpd ,ndpd, dpy, interp=True):
    '''
    interp : Shall we interpolate drom dpd to ndpd. True or False
    dpd : data per day
    ndpd : new data per day
    Returns a disctionnary with stacked time series over the N years of the excel file. Each year is normalized and interpolated (if true)
    '''
    input_data = pd.ExcelFile(path_input_data + input_file)

    df = pd.read_excel(path_input_data + input_file)

    myarray = df.values
    one_d = myarray.ravel()
    one_d = one_d.astype(float)
    mysize = one_d.size
    assert(mysize % dpd == 0), 'import_excel : Data does not cover an integer number of days'

    # Renormalize to 1MW mean for each full year and discard the rest
    dataperyear = dpd*dpy
    nfullyears = int( mysize /(dataperyear) )

    for i in range(nfullyears):
        sublist = one_d[i * dataperyear: (i + 1) * dataperyear]
        mean = np.mean(sublist)
        one_d[i * dataperyear: (i + 1) * dataperyear] = sublist / mean
    one_d = one_d[0:nfullyears * dataperyear]

    # Interpolate on the new grid

    if ndpd is not None:
        ndays = nfullyears*dpy
        oldx = np.arange(0, ndays, 1./dpd)
        newx = np.arange(0, ndays, 1./ndpd)
        newy = np.interp(newx, oldx, one_d)
    else:
        newy = one_d

    return newy