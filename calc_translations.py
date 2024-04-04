import numpy as np
import pickle as pkl
import os
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

def sine_function(Dt):
    x = np.linspace(0, 2*np.pi, Dt, endpoint = False)
    sine = np.sin(x)
    return sine

def translate(data, d):
    while d < 0: #what is the use of this ?
        d = d + data.size
    tmp = np.zeros(data.size)
    for i in range(data.size):
        tmp[i] = data[(i + d) % data.size]
    return tmp

def translate_J(data,d):
    
    return np.roll(data,d)

def calc_residue(data, wavelets, sparse_wavelets):
    '''
    Calculate the baseline residue, before optimizing the position of the wavelet on the signal.

    '''
    data_no_mean = data - np.mean(data)
    betas = lsqr(sparse_wavelets, data_no_mean, damp=0.001, atol=0, btol=0, conlim=0)[0]
    
    for i in range(wavelets.shape[0]):
        data_no_mean = data_no_mean - betas[i]*wavelets[i,:]
    residue = np.sum(data_no_mean*data_no_mean)
    print('Residue')
    print(residue)
    return residue

def calc_residue_j(y_fit, y_ref):
    residuals = y_fit - y_ref
    return residuals

def error_function(shift, data, wavelets, sparse_wavelets):
    shifted_signal = translate_J(data, int(shift) % len(data))
    print(shifted_signal)
    data_no_mean = shifted_signal - np.mean(shifted_signal)
    betas = lsqr(sparse_wavelets, data_no_mean, damp=0.001, atol=0, btol=0, conlim=0)[0]
    
    for i in range(wavelets.shape[0]):
        data_no_mean = data_no_mean - betas[i]*wavelets[i,:]
    residue = np.sum(data_no_mean*data_no_mean)
    print('Residue')
    print(residue)
    return residue
    # return calc_residue(shifted_signal, wavelets, sparse_wavelets)

def calc_best_trans(wavelets, sparse_wavelets, signal_in, ndpd, dpy):
    '''
    signal_in : the signal to find the best translation
    wavelets : signal of the wavelet 1 to -1 in a square (or sine) shape
    sparse_wavelets : same as wavelets but csr matrix transposed 
    '''
    assert(len(signal_in) == ndpd * dpy), 'dimension mismatch'
    veclength = ndpd * dpy
    best_residue = np.sum(signal_in*signal_in)
    best_day = 0
    for counter in range(veclength):
        data = translate(signal_in, counter)
        # print("signal_in")
        # print(signal_in)
        # print('data')
        # print(data)
        residue = calc_residue(data, wavelets, sparse_wavelets)
        if residue < best_residue:
            best_day = counter
            best_residue = residue
            print(counter)
            print(residue)

    return best_day

def calc_best_trans_J(wavelets, sparse_wavelets, signal_in, ndpd, dpy):
    '''
    signal_in : the signal to find the best translation
    wavelets : signal of the wavelet 1 to -1 in a square (or sine) shape
    sparse_wavelets : same as wavelets but csr matrix transposed 
    '''
    # Initialisation du décalage (shift)
    initial_shift = 0

    # Minimisation de l'erreur
    result = minimize(error_function, initial_shift, args=(signal_in, wavelets, sparse_wavelets))
    print(result)
    # Récupération du décalage optimal
    optimal_shift = int(result.x) % len(signal_in)

    # assert(len(signal_in) == ndpd * dpy), 'dimension mismatch'
    # veclength = ndpd * dpy
    # best_residue = np.sum(signal_in*signal_in)
    # best_day = 0
    # for counter in range(veclength):
    #     data = translate(signal_in, counter)
    #     print("signal_in")
    #     print(signal_in)
    #     print('data')
    #     print(data)
    #     residue = calc_residue(data, wavelets, sparse_wavelets)
    #     if residue < best_residue:
    #         best_day = counter
    #         best_residue = residue
    #         print(counter)
    #         print(residue)

    # return best_day
    return optimal_shift

def calculate_all_translations(path_trans, translation_name, 
                               ndpd, dpy, input_data, wl_shape, 
                               recompute_translation= False):
    '''
    Compute best translations for each years of the input data
    :param ndpd: data per day
    :param dpy: days per year
    :param input_data: stacked time series (1D vector)
    :param wl_shape: 'square' or 'sine_function' : shape of the wavelet
    :return: a list of translations for each uear
    '''
    veclength = ndpd*dpy
    Nyears = int(len(input_data)/veclength)

    signal_length = len(input_data)
    assert (signal_length % (dpy * ndpd) == 0), 'The signal length is not an integer number of years.'
    assert(wl_shape != 'sine' or wl_shape != 'square'), 'Shape error. must be either square or sine_function'

    # Check if the file exists and is consistent with the number of years of the input signal. If not, recomputing the translations
    filename_pkl = os.path.join(os.getcwd(), path_trans, 'results_translation_'+ translation_name +'.pkl')

    if os.path.exists(filename_pkl) and not recompute_translation:
        # Load the data from the 'results_translation.pkl' file if its size is consistent with the number of year of the input signal
        with open(filename_pkl, 'rb') as file:
            trans = pkl.load(file)
        if len(trans) == Nyears :
            print(f"Loading existing translation file: {filename_pkl}")
    else:
        # File does not exist, so compute the translation
        print("Computing translation...")

        trans = []
        for k in range(Nyears):
            signal_in = input_data[k*veclength: (k+1)*veclength] #get the data for one year only to perform the translation
            # Year
            # Creat year mother wavelet
            
            Dt = dpy * ndpd
            signal_length = dpy * ndpd
            
            vec_year = np.zeros((1, signal_length))
            if wl_shape == 'square':
                vec_year[0, 0:  Dt // 2] = 1.  # /math.sqrt(Dt)
                vec_year[0, Dt // 2:  Dt] = -1.  # /math.sqrt(Dt)
            if wl_shape == 'sine':
                vec_year[0, :] =  sine_function(Dt)
            vec_year_sparse = sparse.csr_matrix(np.transpose(vec_year))
            # print(vec_year)
            # print(vec_year_sparse)
            best_trans_year = calc_best_trans(vec_year, vec_year_sparse, signal_in, ndpd, dpy)

            # ----------------------------
            # Week
            # Create week mother wavelets
            Dt = 7 * ndpd  # points
            vec_week = np.zeros((52, signal_length))
            c = 0
            i = 0
            # while i < 52:  # loop over the time scales # pourquoi pas une boucle for ?
            #     if wl_shape == 'square':
            #         vec_week[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
            #         vec_week[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            #     if wl_shape == 'sine':
            #         vec_week[c, i*Dt : (i+1)*Dt ] = sine_function(Dt)
            #     c = c + 1
            #     i = i + 1

            for i in range(52): 
                c=i
                if wl_shape == 'square':
                    vec_week[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                    vec_week[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
                if wl_shape == 'sine':
                    vec_week[c, i*Dt : (i+1)*Dt ] = sine_function(Dt)

            vec_week_sparse = sparse.csr_matrix(np.transpose(vec_week))

            best_trans_week = calc_best_trans(vec_week, vec_week_sparse, signal_in, ndpd, dpy)

            # ----------------------------
            # Days
            # Creat daymother wavelets
            Dt = ndpd  # points /day
            vec_day = np.zeros((dpy, signal_length))
            c = 0
            i = 0
            # while i < dpy:  # loop over the time scales
            #     if wl_shape == 'square':
            #         vec_day[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
            #         vec_day[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            #     if wl_shape == 'sine':
            #         vec_day[c, i*Dt : (i+1)*Dt] = sine_function(Dt)
            #     c = c + 1
            #     i = i + 1
            for i in range(dpy):  # loop over the time scales
                if wl_shape == 'square':
                    vec_day[i, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                    vec_day[i, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
                if wl_shape == 'sine':
                    vec_day[i, i*Dt : (i+1)*Dt] = sine_function(Dt)
               
            
            vec_day_sparse = sparse.csr_matrix(np.transpose(vec_day))

            best_trans_day = calc_best_trans(vec_day, vec_day_sparse, signal_in, ndpd, dpy)

            # print([best_trans_day, best_trans_week, best_trans_year])
            print(f"Best translation day = {best_trans_day}")
            print(f"Best translation week = {best_trans_week}")
            print(f"Best translation year = {best_trans_year}")
            
            trans.append( [best_trans_day, best_trans_week, best_trans_year] )

            # Save the results of the translation in the 'translation/' directory
            with open(filename_pkl, 'wb') as file:
                pkl.dump(trans, file)
    return trans




