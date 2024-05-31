import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import pickle

'''
This function :
    - Interpolate on greater number of points
    - Normalize each time series for each year
    - stack all years in a dictionnary : stacked data
For example  Nyears PV data are accessible in stacked_data[Solaire]
'''
def import_excel(path_input_data,input_file, dpd ,ndpd, dpy, interp=True, norm = 'mean'):
    '''
    interp : Shall we interpolate drom dpd to ndpd. True or False
    dpd : data per day
    ndpd : new data per day
    Returns a dictionnary with stacked time series over the N years of the excel file. Each year is normalized and interpolated (if true)
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
        if norm =='mean':
            mean = np.mean(sublist)
        elif norm == 'max':
            mean = np.max(sublist)
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

def optimize_enr(country_name, Load_ts, PV_ts, Wind_ts, mean_load, save_results = False):
    prob = LpProblem(f"myProblem", LpMinimize)
    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = 10 # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000 # MW used for the binary variable, voluntraily very high

    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')


    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += x_pv * PV_ts[t] + ts_dispatchable[t] + x_wind * Wind_ts[t]+p_dech[t] == Load_ts[t] +p_ch[t]+p_curt[t]

    # Constraint 2: storage
    for t in range(1, signal_length):
        prob += SOC_ts[t] == SOC_ts[t-1] + (stock_efficiency*p_ch[t]-p_dech[t])*dt
    
        # Binary variable: can't charge and discharge at the same time
        prob += p_ch[t] <= (1-dech_active[t])*P_max 
        prob += p_dech[t] <= (dech_active[t])*P_max

    #TODO : trouver une meilleure contrainte pour p_stock
    prob+= p_ch[0]==0
    prob+=p_dech[0]==0
    prob += SOC_ts[0] == SOC_ts[signal_length-1] #same state of charge at the start and end

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() == Load_ts.sum()

    # Fonction objectif
    prob += lpSum(ts_dispatchable[t] for t in range(signal_length))*dt

    prob.solve(GUROBI())

    # Afficher les résultats
    print("Status:", LpStatus[prob.status])
    print("Coefficient optimal pour PV:", x_pv.varValue)
    print("Coefficient optimal pour Wind:", x_wind.varValue)


    # Get results of optimization 
    optimized_pv = [x_pv.varValue * PV_ts[t] for t in range(signal_length)]
    optimized_wind = [x_wind.varValue * Wind_ts[t] for t in range(signal_length)]
    optimized_dispatchable = [ts_dispatchable[t].varValue for t in range(signal_length)]
    optimized_stock = [SOC_ts[t].varValue for t in range(signal_length)]
    optimized_p_curt = [p_curt[t].varValue for t in range(signal_length)]
    optimized_charge = [p_ch[t].varValue for t in range(signal_length)]
    optimized_discharge = [p_dech[t].varValue for t in range(signal_length)]


    # Calculate energy totals
    E_wind = np.sum(optimized_wind)*mean_load
    E_pv = np.sum(optimized_pv)*mean_load
    E_dispatch = np.sum(optimized_dispatchable)*mean_load
    E_curt = np.sum(optimized_p_curt)*mean_load
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))*mean_load
    E_stock = np.sum(optimized_charge)*mean_load
    E_destock = np.sum(optimized_discharge)*mean_load

    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts), 
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue*mean_load, 
    'wind_capacity': x_wind.varValue*mean_load,
    'dispatchable_capacity':np.max(optimized_dispatchable)*mean_load, 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock
}
    if save_results :
        filename = f'results/{country_name}/optimization_results.pickle'

        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    optimization_results = pd.read_pickle(f'results/{country_name}/optimization_results.pickle')
    return results
