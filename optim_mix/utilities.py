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
        else:
            mean = 1
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
def optimize_enr_chu(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None,save_results = False):
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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    if save_results :
        filename = f'results/{country_name}/Chu/optimization_results_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}/Chu'):
            os.makedirs(f'results/{country_name}/Chu')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results


def optimize_enr(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None,save_results = False):
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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    if save_results :
        filename = f'results/{country_name}/optimization_results_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_enr_50(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None,save_results = False):
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

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() <= 0.5*Load_ts.sum()

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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    if save_results :
        filename = f'results/{country_name}/optimization_results_{state_name}_50c.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_enr_150(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None,save_results = False):
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

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() <= 1.5*Load_ts.sum()

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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    if save_results :
        filename = f'results/{country_name}/optimization_results_{state_name}_150c.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_enr_no_dispatch(country_name, Load_ts, PV_ts, Wind_ts,I_pv, I_wind, mean_load, save_results = False):
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
        prob+=ts_dispatchable[t]==0
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


    # Fonction objectif : on minimiser les GES liés aux ENR
    prob+=lpSum(I_pv*x_pv*PV_ts.sum()+I_wind*x_wind*Wind_ts.sum())

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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    if save_results :
        filename = f'results/{country_name}/optimization_results_0_dispatch_bis.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_enr_budget(country_name, Load_ts, PV_ts, Wind_ts,I_pv, I_wind, budget, mean_load, save_results = False):
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
        prob+=ts_dispatchable[t]==0
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

    prob += I_pv*x_pv*PV_ts.sum()+I_wind*x_wind*Wind_ts.sum() <= budget
    # Fonction objectif : on minimiser l'énergie pilotable
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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
    }
    if save_results :
        filename = f'results/{country_name}/optimization_results_carbon_budget.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results
def optimize_enr_capacity(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None, save_results = False):
    prob = LpProblem(f"myProblem", LpMinimize)
    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = 10 # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000000 # MW used for the binary variable, voluntraily very high
    
    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    max_dispatchable = LpVariable('Dispatchable_capaicity', lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')


    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += ts_dispatchable[t]<= max_dispatchable
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

    # Fonction objectif : on minimiser les GES liés aux ENR
    prob+=max_dispatchable

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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    
    if save_results :
        filename = f'results/{country_name}/optimization_results_capacity_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results


def optimize_enr_capacity(country_name, Load_ts, PV_ts, Wind_ts, mean_load, state_name = None, save_results = False):
    prob = LpProblem(f"myProblem", LpMinimize)
    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = 10 # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000000 # MW used for the binary variable, voluntraily very high
    
    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    max_dispatchable = LpVariable('Dispatchable_capaicity', lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')


    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += ts_dispatchable[t]<= max_dispatchable
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

    # Fonction objectif : on minimiser les GES liés aux ENR
    prob+=max_dispatchable

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

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
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
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    
    if save_results :
        filename = f'results/{country_name}/optimization_results_capacity_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def format_load_data(country_name, state_name = None):
    if not os.path.exists(f'../input_time_series/{country_name}/'):
        os.makedirs(f'../input_time_series/{country_name}/')
    if state_name:
        file_path = f'../input_time_series/{country_name}/{country_name}_{state_name}_demand_Plexos_2015.xlsx'
    else: 
        file_path = f'../input_time_series/{country_name}/{country_name}_demand_Plexos_2015.xlsx'

    if os.path.exists(file_path):
        pass
    else: 
        country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
        data = pd.read_csv('../input_time_series/All Demand UTC 2015.csv', index_col =0)
        iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
        if state_name:
            print(data.columns.str.endswith(state_name))
            column_name = data.columns[data.columns.str.endswith(state_name)].item()
            data[column_name].to_excel(file_path, index =False)
        else:
            column_name = data.columns[data.columns.str.endswith(iso_code)].item()
            data[column_name].to_excel(file_path, index =False)
        
    return file_path.split('/',2)[-1]

def optimize_enr_budget_constraint(country_name, Load_ts, PV_ts, Wind_ts,I_pv, I_wind,budget, mean_load, state_name = None, save_results = False): 
    '''
    This function optimizes the electricity mix to minimize dispatchable energy under a carbon budget constraint on installed capacities of renewable energy.
    - I_pv : carbon intensity of PV (ktCO2eq/MW)
    - I_wind : carbon intensity of a wind turbines (ktCO2eq/MW)
    - budget : carbon budget allocated to the ountry for electricity generation (ktCO2eq)

    Returns: 
    x_pv and x_wind the capacities to install for PV and wind energy. 
    '''
    
    prob = LpProblem(f"myProblem_{country_name}", LpMinimize)
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

    prob += x_pv*I_pv+x_wind*I_wind <= budget

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
    E_wind = np.sum(optimized_wind)
    E_pv = np.sum(optimized_pv)
    E_dispatch = np.sum(optimized_dispatchable)
    E_curt = np.sum(optimized_p_curt)
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))
    E_stock = np.sum(optimized_charge)
    E_destock = np.sum(optimized_discharge)

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue, 
    'wind_capacity': x_wind.varValue,
    'dispatchable_capacity':np.max(optimized_dispatchable), 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    
    if save_results :
        filename = f'results/{country_name}/optimization_results_budget_constraint_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results


def optimize_enr_budget_constraint_kWh(country_name, Load_ts, PV_ts, Wind_ts,I_pv, I_wind,budget, mean_load, state_name = None, save_results = False): 
    '''
    This function optimizes the electricity mix to minimize dispatchable energy under a carbon budget constraint on installed capacities of renewable energy.
    - I_pv : carbon intensity of PV (ktCO2eq/MW)
    - I_wind : carbon intensity of a wind turbines (ktCO2eq/MW)
    - budget : carbon budget allocated to the ountry for electricity generation (ktCO2eq)

    Returns: 
    x_pv and x_wind the capacities to install for PV and wind energy. 
    '''
    
    prob = LpProblem(f"myProblem_{country_name}", LpMinimize)
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

    prob += x_pv*PV_ts.sum()*I_pv+x_wind*Wind_ts.sum()*I_wind <= budget

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
    E_wind = np.sum(optimized_wind)
    E_pv = np.sum(optimized_pv)
    E_dispatch = np.sum(optimized_dispatchable)
    E_curt = np.sum(optimized_p_curt)
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))
    E_stock = np.sum(optimized_charge)
    E_destock = np.sum(optimized_discharge)

    country_codes = pd.read_csv('../countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue, 
    'wind_capacity': x_wind.varValue,
    'dispatchable_capacity':np.max(optimized_dispatchable), 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch),
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch),
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)
}
    
    if save_results :
        filename = f'results/{country_name}/optimization_results_budget_constraint_{state_name}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results
