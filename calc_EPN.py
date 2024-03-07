import numpy as np

def calc_epn(beta, satisfactions, time_scales, dpy, load_factor, shape='square'):
    '''
    Calculate Energy, Power, nb of cycles from input betas
    Returns E, P, N, Usage factor for diferent satisfaction rates
    - load_factor : average power, the mean energy consumption
    :param shape:
    '''
    Nyears = len(beta[-1])

    print('Calculations are made on a '+   str(Nyears) + ' years dataset')
    #Indices : length, satisfaction
    pmax = np.zeros((len(time_scales), len(satisfactions))) # Power
    emax = np.zeros((len(time_scales), len(satisfactions))) # Energy
    n =    np.zeros((len(time_scales), len(satisfactions))) # Number of cycles
    uf =   np.zeros((len(time_scales), len(satisfactions))) # Usage factor
    serv = np.zeros((len(time_scales), len(satisfactions))) # Service

    for i in range(len(time_scales)):
        veclength = time_scales[i]
        betac = beta[i]  # Consumption power
        betac_array = np.array(betac)
        # Calculation of the number of instants to satisfy in order to be > x%

        isatis = np.ceil(np.asarray(satisfactions) * len(betac) / 100.) - 1.
        isatis = isatis.astype(int)

        dech_satis = np.zeros(len(satisfactions))
        n_satis = np.zeros(len(satisfactions))
        # Discharge power needed once the production has been resized
        for s in range(len(satisfactions)):
            # Calculate the discharge power needed to satisfy x% of the instants
            dech = np.sort(np.abs(betac_array))
            dech_satis[s] = dech[isatis[s]]
            n_satis[s] = sum(np.minimum(dech, dech_satis[s])) / dech_satis[s] / Nyears

        pmax[i, :] = dech_satis * load_factor
        if shape is 'sine':
            emax[i, :] = pmax[i, :] * veclength / 2. *2./np.pi
        else:
            emax[i, :] = pmax[i, :] * veclength / 2.
        n[i, :] = n_satis
        uf[i, :] = 100. * n[i, :] * veclength / (dpy*24)
        serv[i, :] = emax[i, :] * n[i, :]
    return {'pmax': pmax, 'emax': emax, 'uf': uf, 'n': n, "serv": serv}