import numpy as np

def calc_epn(beta, satisfactions, time_scales, dpy, load_factor):
    '''
    Calculate Energy, Power, nb of cycles from input betas
    Returns E, P, N, Usage factor for diferent satisfaction rates
    - load_factor : average power, the mean energy consumption
    '''
    Nyears = beta[-1].size
    #Indices : length, satisfaction
    pmax = np.zeros((len(time_scales), len(satisfactions))) # Power
    emax = np.zeros((len(time_scales), len(satisfactions))) # Energy
    n =    np.zeros((len(time_scales), len(satisfactions))) # Number of cycles
    uf =   np.zeros((len(time_scales), len(satisfactions))) # Usage factor
    serv = np.zeros((len(time_scales), len(satisfactions))) # Service

    for i in range(len(beta)):
        veclength = time_scales[i]
        betac = beta[i]  # Consumption power
        # Calculation of the number of instants to satisfy in order to be > x%

        isatis = np.ceil(np.asarray(satisfactions) * betac.size / 100.) - 1.
        isatis = isatis.astype(int)

        dech_satis = np.zeros(len(satisfactions))
        n_satis = np.zeros(len(satisfactions))
        # Discharge power needed once the production has been resized
        for s in range(len(satisfactions)):
            # Calculate the discharge power needed to satisfy x% of the instants
            dech = np.sort(abs(betac))
            dech_satis[s] = dech[isatis[s]]
            n_satis[s] = sum(np.minimum(dech, dech_satis[s])) / dech_satis[s] / Nyears

        pmax[i, :] = dech_satis * load_factor
        emax[i, :] = pmax[i, :] * veclength / 2.
        n[i, :] = n_satis
        uf[i, :] = 100. * n[i, :] * veclength / (dpy*24)
        serv[i, :] = emax[i, :] * n[i, :]
    return {'pmax': pmax, 'emax': emax, 'uf': uf, 'n': n, "serv": serv}