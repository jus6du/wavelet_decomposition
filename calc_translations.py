import numpy as np
import pickle as pkl
from scipy import sparse
from scipy.sparse.linalg import lsqr


def sine_function(Dt):
    x = np.linspace(0, 2*np.pi, Dt, endpoint = False)
    sine = np.sin(x)
    return sine

def translate(data, d):
    while d < 0:
        d = d + data.size
    tmp = np.zeros(data.size)
    for i in range(data.size):
        tmp[i] = data[(i + d) % data.size]
    return tmp


def calc_residue(data, wavelets, sparse_wavelets):
    data = data - np.mean(data)
    betas = lsqr(sparse_wavelets, data, damp=0.001, atol=0, btol=0, conlim=0)[0]
    for i in range(wavelets.shape[0]):
        data = data - betas[i]*wavelets[i,:]
    residue = np.sum(data*data)
    return residue

def calc_best_trans(wavelets, sparse_wavelets, signal_in, ndpd, dpy):
    assert(len(signal_in) == ndpd * dpy), 'dimension mismatch'
    veclength = ndpd * dpy
    best_residue = np.sum(signal_in*signal_in)
    best_day = 0
    for counter in range(veclength):
        data = translate(signal_in, counter)
        residue = calc_residue(data, wavelets, sparse_wavelets)
        if residue < best_residue:
            best_day = counter
            best_residue = residue
            print(counter)
            print(residue)

    return best_day

def calc_trans(ndpd, dpy, input_data, wl_shape):
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
    assert(dpy*ndpd % len(input_data) ),'Number of years and points are not consistent'
    assert(wl_shape != 'sine' or wl_shape != 'square'), 'Shape error. must be either square or sine_function'
    trans = []
    for k in range(Nyears):
        signal_in = input_data[k*veclength: (k+1)*veclength]
        # Year
        # Creat year mother waveley
        #
        Dt = dpy * ndpd
        signal_length = dpy * ndpd
        #
        vec_year = np.zeros((1, signal_length))
        if wl_shape == 'square':
            vec_year[0, 0:  Dt // 2] = 1.  # /math.sqrt(Dt)
            vec_year[0, Dt // 2:  Dt] = -1.  # /math.sqrt(Dt)
        if wl_shape == 'sine':
            vec_year[0, :] =  sine_function(Dt)
        vec_year_sparse = sparse.csr_matrix(np.transpose(vec_year))

        best_trans_year = calc_best_trans(vec_year, vec_year_sparse, signal_in, ndpd, dpy)

        # ----------------------------
        # Week
        # Creat week mother wavelets
        Dt = 7 * ndpd  # points
        vec_week = np.zeros((52, signal_length))
        c = 0
        i = 0
        while i < 52:  # loop over the time scales
            if wl_shape == 'square':
                vec_week[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                vec_week[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            if wl_shape == 'sine':
                vec_week[c, i*Dt : (i+1)*Dt ] = sine_function(Dt)
            c = c + 1
            i = i + 1

        vec_week_sparse = sparse.csr_matrix(np.transpose(vec_week))

        best_trans_week = calc_best_trans(vec_week, vec_week_sparse, signal_in, ndpd, dpy)

        # ----------------------------
        # Days
        # Creat daymother wavelets
        Dt = ndpd  # points /day
        vec_day = np.zeros((dpy, signal_length))
        c = 0
        i = 0
        while i < dpy:  # loop over the time scales
            if wl_shape == 'square':
                vec_day[c, 2 * i * Dt // 2: (2 * i + 1) * Dt // 2] = 1.
                vec_day[c, (2 * i + 1) * Dt // 2: (2 * i + 2) * Dt // 2] = -1.
            if wl_shape == 'sine':
                vec_day[c, i*Dt : (i+1)*Dt] = sine_function(Dt)
            c = c + 1
            i = i + 1
        vec_day_sparse = sparse.csr_matrix(np.transpose(vec_day))

        best_trans_day = calc_best_trans(vec_day, vec_day_sparse, signal_in, ndpd, dpy)

        print([best_trans_day, best_trans_week, best_trans_year])
        trans.append( [best_trans_day, best_trans_week, best_trans_year] )
    return trans


def load_trans(path_trans, trans_file, stacked_input_data, signal_type, ndpd, dpy, shape, do_calc = True):
    '''
        This function either load or recompute translations and save them.
        Calculation could take quite some time, go and grab a coffee
    :param path_trans: whe translations are saved
    :param trans_file: name of the saved / imported translation file
    :param stacked_input_data: input_data
    :param signal_type: 'Consommation', 'Eolien',...
    :param ndpd: data per day
    :param dpy: day per year
    :param shape: 'square' or 'sine_function'
    :param do_calc: if Truen compute new translations. If false, import
    :return:
    '''

    if do_calc:
        print('Computing translations began...')
        trans = calc_trans(ndpd, dpy, stacked_input_data[signal_type], wl_shape=shape)
        pkl.dump(trans, open(path_trans + trans_file + '.p', "wb") )
        print('Computing translations ended')
    else:
        print('Importing translations')
        trans = pkl.load(open(path_trans + trans_file + '.p', "rb"))
    return trans



# ----------------------------
# ----- Old translation script
# ----------------------------

# def simple_calc_beta(data, vector_length):
#     assert(vector_length % 2 == 0), 'simple_calc_beta : vector length is not a multiple of 2'
#     betas = []
#     i0 = 0
#     nbp = int(vector_length / 2)
#     while i0 + vector_length <= data.size:
#         integ = 0.
#         for i in range(i0, i0 + nbp):
#             integ = integ + data[i]
#         for i in range(i0 + nbp, i0 + 2 * nbp):
#             integ = integ - data[i]
#         integ = integ / vector_length
#         for i in range(i0, i0 + nbp):
#             data[i] = data[i] - integ
#         for i in range(i0 + nbp, i0 + 2 * nbp):
#             data[i] = data[i] + integ
#         betas.append(integ)
#         i0 = i0 + int(vector_length)
#     return betas
#
# def simple_calc_beta_sqrt(data, vector_length):
#     assert(vector_length % 2 == 0), 'simple_calc_beta : vector length is not a multiple of 2'
#     betas = []
#     i0 = 0
#     nbp = int(vector_length / 2)
#     vec_year
#     # while i0 + vector_length <= data.size:
#     #     integ = 0.
#     #     for i in range(i0, i0 + nbp):
#     #         integ = integ + data[i]
#     #     for i in range(i0 + nbp, i0 + 2 * nbp):
#     #         integ = integ - data[i]
#     #     integ = integ / vector_length
#     #     for i in range(i0, i0 + nbp):
#     #         data[i] = data[i] - integ
#     #     for i in range(i0 + nbp, i0 + 2 * nbp):
#     #         data[i] = data[i] + integ
#     #     betas.append(integ)
#     #     i0 = i0 + int(vector_length)
#     return betas
#
# def calc_translation(data, dataperday,dpy):
#     '''
#     This function compute the best day, week and year translations for each year
#     '''
#     assert(data.size == dpy * dataperday), 'calc_translation : data size should be one year'
#
#     # Year
#     best_d_year = 0
#     best_residue = np.sum(data * data)
#     for d in range(dataperday * dpy // 2):
#         tmpdata = translate(data, d)
#         simple_calc_beta(tmpdata, dataperday * dpy)
#         residue = np.sum(tmpdata*tmpdata)
#         if residue < best_residue:
#             best_residue = residue
#             best_d_year = d
#
#     # Week
#     best_d_week = 0
#     best_residue = np.sum(data*data)
#     for d in range(dataperday * 7 // 2):
#         tmpdata = translate(data, d)
#         simple_calc_beta(tmpdata, dataperday * 7)
#         residue = np.sum(tmpdata*tmpdata)
#         if residue < best_residue:
#             best_residue = residue
#             best_d_week = d
#
#     # Day
#     best_d_day = 0
#     best_residue = np.sum(data*data)
#     for d in range(dataperday // 2):
#         tmpdata = translate(data, d)
#         simple_calc_beta(tmpdata, dataperday)
#         residue = np.sum(tmpdata*tmpdata)
#         if residue < best_residue:
#             best_residue = residue
#             best_d_day = d
#
#     return [best_d_day, best_d_week, best_d_year]
