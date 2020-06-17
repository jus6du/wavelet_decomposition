import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle as pkl
from scipy import sparse
from scipy.sparse.linalg import lsqr
import xlsxwriter

from calc_translations import translate

'''
This function generation a matric with square shape wavelets
'''
def generate_square_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                              trans_vec,
                              path_matrix, matrix_name,
                              import_matrix = True):
    #############
    # Translations
    #############
    [transday, transweek, transyear] = trans_vec

    #############
    #base vectors
    #############
    signal_length = dpd*dpy
    if os.path.exists(path_matrix + matrix_name) and import_matrix:
        A_sparse  = sparse.load_npz(path_matrix + matrix_name)
#         A = sparse.csr_matrix.todense(A_sparse)
#         A = np.asarray(A)
        A = [] # only needA_ sparse for the lsqr algorith
        print('Importing matrix A square')
    else:
        print('Computing Matrix A square')

        ###############
        ## Create wavelets Phi and matrix A
        ################
        Phi0 = np.ones((1, signal_length)) / math.sqrt(dpy * dpd)
        ##
        Dt = dpy* dpd # points
        Phi1 = np.zeros(((2**vecNb_yr-1) , signal_length ))
        c = 0
        for k in range(vecNb_yr): # loop over the subdivisions
            i=0
            while i < 2**k: # loop over the time scales
                Phi1[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/math.sqrt(Dt)
                Phi1[c, (2*i+1)* Dt//2 : (i+1)* Dt ] = -1.#/math.sqrt(Dt)
                Phi1[c,:] = translate(Phi1[c,:], -transyear)
                c = c +1
                i= i + 1

            Dt =Dt // 2

        ## Phi2 seconde set of wavelets
        #
        Dt = 7*dpd # points
        Phi2 = np.zeros((52*(2**vecNb_week-1), signal_length))
        c = 0
        for k in range(vecNb_week): # loop over the subdivisions
            i=0
            while i < 2**k*52: # loop over the time scales
                Phi2[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/ math.sqrt(Dt)
                Phi2[c, (2 * i + 1) * Dt // 2: (i +1) * Dt] = -1.# / math.sqrt(Dt)
                Phi2[c,:] = translate(Phi2[c,:], -transweek)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        ## Phi3 seconde set of wavelets
        #
        Dt = dpd # points /day
        Phi3 = np.zeros((dpy*(2**vecNb_day-1) ,signal_length ))
        c = 0
        for k in range(vecNb_day): # loop over the subdivisions
            i=0
            while i < 2**k*dpy: # loop over the time scales
                Phi3[c, 2*i*Dt//2 : (2*i+1)* Dt//2 ] = 1.#/ math.sqrt(Dt)
                Phi3[c, (2 * i + 1) * Dt // 2: (i + 1) * Dt] = -1.# / math.sqrt(Dt)
                Phi3[c,:] = translate(Phi3[c, :], -transday)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        A = np.transpose(np.concatenate((Phi0, Phi1, Phi2,Phi3)))

        A_sparse = sparse.csr_matrix(A)
#         Making sure that A is normnalized
        epsilon = 10e-5
#         assert(max(np.sum(np.square(A), axis = 0))-1. < epsilon and 1. - min(np.sum(np.square(A), axis = 0)) < epsilon), 'wavelets are not properly normalized'
        sparse.save_npz(path_matrix + matrix_name, sparse.csr_matrix(A))
    return A_sparse


def sine_function(Dt):
    x = np.linspace(0, 2*np.pi, Dt,endpoint=False)
    sine = np.sin(x)
    return sine

def generate_sine_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                              trans_vec,
                              path_matrix, matrix_name,
                              import_matrix = True):
    '''
    This function generation a matric with sine shape wavelets
    '''
    #############
    # Translations
    #############
    [transday, transweek, transyear] = trans_vec

    #############
    #base vectors
    #############
    signal_length = dpd*dpy
    if os.path.exists(path_matrix + matrix_name) and import_matrix:
        A_sparse  = sparse.load_npz(path_matrix + matrix_name)
#         A = sparse.csr_matrix.todense(A_sparse)
#         A = np.asarray(A)
        A = [] # only needA_ sparse for the lsqr algorith
        print('Importing matrix A sine')
    else:
        print('Computing Matrix A sine')

        ###############
        ## Create wavelets Phi and matrix A
        ################
        Phi0 = np.ones((1, signal_length))# / math.sqrt(dpy * dpd)
        ##
        Dt = dpy* dpd # points
        Phi1 = np.zeros(((2**vecNb_yr-1) , signal_length ))
        c = 0
        for k in range(vecNb_yr): # loop over the subdivisions
            i=0
            while i < 2**k: # loop over the time scales
                Phi1[c, 2*i*Dt//2 : (2*i+2)* Dt//2 ] =  sine_function(Dt)
                Phi1[c,:] = translate(Phi1[c,:], -transyear)
                c = c +1
                i= i + 1

            Dt =Dt // 2

        ## Phi2 seconde set of wavelets
        #
        Dt = 7*dpd # points
        Phi2 = np.zeros((52*(2**vecNb_week-1), signal_length))
        c = 0
        for k in range(vecNb_week): # loop over the subdivisions
            i=0
            while i < 2**k*52: # loop over the time scales
                Phi2[c, i*Dt : (i+1)* Dt ] =  sine_function(Dt)
                Phi2[c,:] = translate(Phi2[c,:], -transweek)
                c = c +1
                i= i + 1
            Dt = Dt // 2

        ## Phi3 seconde set of wavelets
        #
        Dt = dpd # points /day
        Phi3 = np.zeros((dpy*(2**vecNb_day-1) ,signal_length ))
        c = 0
        for k in range(vecNb_day): # loop over the subdivisions
            i=0
            if Dt <= 4:
                while i < 2 ** k*dpy:  # With 4 or two points cannot create a sinus
                    Phi3[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.  # /math.sqrt(Dt)
                    Phi3[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.  # /math.sqrt(Dt)
                    Phi3[c, :] = translate(Phi3[c, :], -transday)
                    c = c + 1
                    i = i + 1

                Dt = Dt // 2
            else:
                while i < 2**k*dpy: # loop over the time scales
                    Phi3[c, i*Dt : (i+1)* Dt ] =  sine_function(Dt)
                    Phi3[c, :] = translate(Phi3[c, :], -transday)
                    c = c +1
                    i= i + 1

                Dt = Dt // 2

        A = np.transpose(np.concatenate((Phi0, Phi1, Phi2,Phi3)))

        A_sparse = sparse.csr_matrix(A)
#         Making sure that A is normnalized
        epsilon = 10e-5
#         assert(max(np.sum(np.square(A), axis = 0))-1. < epsilon and 1. - min(np.sum(np.square(A), axis = 0)) < epsilon), 'wavelets are not properly normalized'
        sparse.save_npz(path_matrix + matrix_name, sparse.csr_matrix(A))
    return A_sparse


def beta_decomposition(A_sparse, signal_in):
    # A_sparse = sparse.csr_matrix(A)
    beta_lsqr = lsqr(A_sparse, signal_in, damp=0.001, atol=0, btol=0, conlim=0)[0]
    # Damping coefficient has to be smaller than 0.1. when damp gets big, we loose the reconstruction ( from damp=0.1). When too small, we loose linearity
    return beta_lsqr



def compute_betas(time_series,  stacked_data,
                 vecNb_yr, vecNb_week, vecNb_day, dpy, dpd, years,
                 trans,
                 path_matrix,
                 beta_path, wl_shape, imp_matrix = True ):
    '''
    This function:
    - Compute betas for each imput signal
    - Reshape betas from a 1D-array to a dictionnary with N (15) time scales rows
    - Translate in the othert directions the beta
    - Export in an excel document with different sheets
    - Concatenate all years in a signle sheet
    - Export concatenated betas in a disctionnary, with input signals as jeys of the dictionnary
    - wl_shape : takes 2 values, either square ore sine
    '''
    #
    signal_length = dpy * dpd

    stacked_betas = {}
    workbook2 = xlsxwriter.Workbook(beta_path + 'betas_stacked.xlsx') #All years are stacked in this excel file. One sheet per input siganm

    saved_sheets = {}
    for signal_type in time_series:

        signal_in = stacked_data[signal_type]
#
# 1) ----- Compute betas for a given input signal -------
# -------- returns a 1D array with N years stacked
        betas = []
        for i, year in enumerate(years):
            matrix_name = "A_"+ year + ".npz"
            print(path_matrix + matrix_name)
            if wl_shape == 'square':
                A_sparse = generate_square_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                    trans[i],
                                                    path_matrix, matrix_name,
                                                    import_matrix = imp_matrix)
                print('Square sparsee matrix or year '+ year +' has been imported')
            elif wl_shape == 'sine':
                A_sparse = generate_sine_wl_matrix(vecNb_yr, vecNb_week, vecNb_day, dpy, dpd,
                                                    trans[i],
                                                    path_matrix, matrix_name,
                                                    import_matrix = imp_matrix)
                print('Sine sparse matrix or year '+ year +' has been imported')
            else:
                print('The type of wavelet is not defined. Please type "square" or "sine"')

            betas.append(beta_decomposition(A_sparse, signal_in[signal_length*i:signal_length*(i+1)]) )

        #
        # -------- Open Excel file ----------
        workbook = xlsxwriter.Workbook(beta_path + 'betas_'+ signal_type + '.xlsx')
        row = 0
        saved_sheets[signal_type] = {}
#
# 2) ----- Reshape betas in a list of 16 time scales -------
# -------- Time scales icludes the offset value
        for i,beta in enumerate(betas):
            saved_sheets[signal_type][years[i]] = []

            worksheet = workbook.add_worksheet(str(years[i]))

            # -- Initialization --
            len_max = dpy *(2**vecNb_day-1)
            newsize = dpy *(2**vecNb_day-1)
            total_vec = vecNb_day+vecNb_week+vecNb_yr # number of time scales
            sheet = []

            beta_offset =[beta[0]]
            beta_year = beta[1 : 1+ 2**vecNb_yr-1] # all betas comming from the yearly motheer wavelet
            beta_week = beta[1+ 2**vecNb_yr - 1: 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1)]
            beta_day = beta[ 1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1):  1+ 2**vecNb_yr-1 + 52*(2**vecNb_week-1) + dpy * (2**vecNb_day-1)]
            #

            sheet.append(beta_offset)
            # --- Year vec ------
            for k in range(vecNb_yr):
                sheet.append(beta_year[2 ** k - 1: 2 ** (k + 1) - 1].tolist())
            # --- Week vec ------
            for k in range(vecNb_week):
                sheet.append(beta_week[52 * (2 ** k - 1): 52 * (2 ** (k + 1) - 1)].tolist())
            # --- Day vec ------
            for k in range(vecNb_day):
                sheet.append(beta_day[dpy * (2 ** k - 1):dpy * (2 ** (k + 1) - 1)].tolist())


            #reverse the list order
            sheet = [sh[::-1] for sh in reversed(sheet) ]

            saved_sheets[signal_type][years[i]] = sheet

            for col, data in enumerate(sheet):
                worksheet.write_column(row, col, data)

            if len(saved_sheets[signal_type][years[i]][-1])>1:
                print('error1')

        workbook.close()
#
# 3) ----- Stack all betas in a 16 dimensions time scale list  -------

        worksheet2 = workbook2.add_worksheet(signal_type)
        row = 0
        # Initialization
        stacked_sheet = [None] * len(saved_sheets[signal_type][years[0]])

        for ts in range(len(stacked_sheet)):
            tmp = []
            for i in range(len(years)):
                tmp.extend(saved_sheets[signal_type][years[i]][ts])

            stacked_sheet[ts] = tmp

        for col, data in enumerate(stacked_sheet):
            worksheet2.write_column(row, col, data)

        stacked_betas[signal_type] = stacked_sheet
        #
    workbook2.close()

    return stacked_betas, saved_sheets




def preplotprocessing(vecNb_yr, vecNb_week , vecNb_day, ndpd, dpy,
                    signal_type, year, years, saved_sheets, do_trans = None ):
    '''
    Preprocess waveley sheets for plot_betas_heatmap() function
This function takes as imputs:
    - saved_sheets
    - signal_type : 'Consommation' or 'Eolien'...
    - list of years included. e.g. ['2012', '2013',...]
    - year : e.g. '2012'
    - if translate is None, there is no translation. else, translating results

What it does:
    - Reshape betas as a dataframe with row of equal size. Eeady to pe ploted
    '''
    #
    # translation
    assert(years[years.index(year)] == year), 'Index error between the translation year and the data'
    if do_trans is None:
            [transday, transweek, transyear] = [0,0,0]
    else:
            [transday, transweek, transyear] = do_trans[years.index(year)]

    #
    # Initialization
    Nb_vec = vecNb_yr + vecNb_week + vecNb_day
    max_nb_betas = dpy*ndpd//2

    assert(max_nb_betas == len(saved_sheets[signal_type][year][0])), 'Inconsistant number of coefficient betas'
    assert(Nb_vec+1 == len(saved_sheets[signal_type][year]) ), 'There is not the right number of time scales' # +1 stands for the offset value
    # Create an empty DataFrame (nan)
    df = pd.DataFrame(np.nan, index=range(Nb_vec), columns=range(max_nb_betas)).transpose()
    # Re-shape
    counter = 0
    # ---- Day vectors
    for k in range(vecNb_day):
        old_vec =saved_sheets[signal_type][year][k]
        oldsize = len(old_vec)
        new_vec = []
        step = int(2**k)
        for i in range(oldsize):
            new_vec = new_vec + [old_vec[i]]*step
        #
        new_vec = translate(np.array(new_vec), transday)

        df[k] = pd.DataFrame({'betas':new_vec})
        counter = counter + 1

    # ------ Week vectors
    # The is 52*7=364 "days" => week wavelets are not covering the full year
    newsize = 364.*ndpd
    for l in range(vecNb_week):
        k = counter
        old_vec = saved_sheets[signal_type][year][k]
        oldsize = len(old_vec)
        new_vec = []
        step = int(newsize/oldsize/2)
        for i in range(oldsize):
            new_vec = new_vec + [old_vec[i]]*step
        #
        new_vec = translate(np.array(new_vec), transweek)

        df[k] = pd.DataFrame({'betas':new_vec})
        counter = counter + 1

    # ------ Year vectors
    for m in range(vecNb_yr):
        k = counter
        old_vec =saved_sheets[signal_type][year][k]
        oldsize = len(old_vec)
        new_vec = []
        step = int(dpy*2**m)
        for i in range(oldsize):
            new_vec = new_vec + [old_vec[i]]*step
        #
        new_vec = translate(np.array(new_vec), transyear)

        df[k] = pd.DataFrame({'betas':new_vec})
        counter = counter + 1

    return df


def stack_betas(saved_sheets, time_series, chosen_years):
    '''
    This function returns stacked betas for chosen number of years
    It takes as arguments :
    - saved_sheets: a dictionnary with signal types as keyx (e.g. "Consommation") and year then
    For instance saved_sheets['Consommation']["2012"] returns a list of N time scales
    - chosen years: the years picked up amoung the "years" imported
    '''
    stacked_betas = {}
    for signal_type in time_series:
        stacked_sheet = [None] * len(saved_sheets[signal_type][chosen_years[0]])

        for ts in range(len(stacked_sheet)):
            tmp = []
            for yr in chosen_years:
                tmp.extend(saved_sheets[signal_type][yr][ts])

            stacked_sheet[ts] = tmp
        print(signal_type)

        stacked_betas[signal_type] = stacked_sheet
    return stacked_betas


def reconstruct(time_scales, reconstructed_time_scales,
                matrix, beta_sheet, title,
                xmin=0, xmax=365,
                dpy=365, dpd=64,
                add_offset=True):
    '''
    This function reconstruct time series for given time scales. It only workd for 1 year signals.
    Takes as inputs :
     - time_scales :  [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380., 8760.] # cycles length, in hours
     - reconstructed_time_scales : e.g. [24] if you want to reconstructa signal filtered with the daily wavelets
     - matrix, square or sine, with the year
     - beta_sheets : A one year list of betas ordered by time scales
     - title : of the figure
     - dpy : days per year. 365 default value
     - dpd : data per day: 64 defauld value
     - add_offset : If you want to add or remove the offset of the siganl
    '''

    #     time_scales = [0.75, 1.5, 3., 6., 12, 24., 42., 84., 168., 273.75, 547.5, 1095., 2190., 4380., 8760.] # cycles length, in hours
    # reconstructed_time_scales = time_scales
    # Concat time scales
    concat_betas = []
    for i, ts in enumerate(time_scales):
        if ts in reconstructed_time_scales:
            concat_betas.extend(beta_sheet[i])
        else:
            concat_betas.extend([0.] * len(beta_sheet[i]))

    if add_offset:
        concat_betas.extend(beta_sheet[-1])

    # PLots options
    sns.set()
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")  # set colors palettte
    #
    time = np.linspace(0, dpy, dpy * dpd)
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    plt.plot(time, np.dot(matrix, concat_betas[::-1]))
    plt.xlim(xmin, xmax)
    plt.xlabel('Days')
    plt.ylabel('Power')
    plt.title(title)
    plt.show()
