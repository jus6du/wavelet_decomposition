import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import os
import pickle as pkl
from scipy import sparse
from scipy.sparse.linalg import lsqr
import scipy.fftpack
import xlsxwriter
import matplotlib.ticker as mticke
from wavelet_decomposition import preplotprocessing

def plot_betas_heatmap(df, signal_type, year , ndpd,
                      cmin = None, cmax= None, ccenter = None):
    '''
    Here we plot the absolute value of betas dataframe
    '''

    # Plot aesthetic settings
    sns.set()
    sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")  # set colors palettte

    plt.rc('font', family='serif')
    # Plot
    # Tick axis definition
    time_scales = ['0.75 ', '1.5', '3 ', '6 ', '12', 'day', '42', '84', 'week', '273.75', '547.5',
            '1095', '2190', '4380', 'year']
    y = [0.5, 1.5 ,2.5 ,3.5 ,4.5 ,5.5 ,6.5 ,7.5 ,8.5 ,9.5 ,10.5,11.5 ,12.5 ,13.5 ,14.5 ]
    #
    time = ["January, 1"] + [str(x) for x in range(30,360,30)]+["December, 31"]
    x = [x*ndpd for x in range(0,360,30)]+[365.*ndpd]
    #
    # Figure settings
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    # Z = (np.absolute(df)).transpose()
    Z = df.transpose()
    ax = sns.heatmap( Z ,
    #                norm=LogNorm(vmin=Z.min().max(), vmax=Z.max().min()),
    #                cmap = "YlOrRd",
                   cmap='coolwarm',
                   center = ccenter,
    #                robust=True, # is supposed to improve contrast
                   vmin= cmin,
                   vmax= cmax,
                   cbar = False
    )
    cbar = ax.figure.colorbar(ax.collections[0])
    # cbar.set_ticks([0.005, 0.085])
    # cbar.set_ticklabels(["Low", "High"])
    cbar.set_label('Charge - Discharge power ')
    ax.set_yticks(y)
    ax.set_yticklabels(time_scales, minor=False, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(time, minor=False, rotation=0)
    plt.ylabel('Storage time scale (hours)', fontsize=20, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=20, fontweight='bold')
    plt.title('Wavelet transform of the signal "'+ signal_type +'" in ' + year , fontsize=20, fontweight='bold')
    plt.ylim(15,0)



    plt.show(block=False)
    plt.pause(3)
    plt.close()

def fft(ndpd, dpy,
        signal, year,
       input_data):
    '''
    Fast Fourrier Transform of input_data
    '''
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.})

    signal_length = dpy * ndpd
    # -----
    # Number of samplepoints
    N = len(input_data)
    # sample spacing
    T = 1.0 / N
    x = np.linspace(0, int(N*T), N)
    y = input_data - np.mean(input_data)
    yf = np.absolute(scipy.fftpack.fft(y))
    xf = 8760./np.linspace(0, 1.0/(2.0*T), int(N/2))

    xcoords = [1, 12 ,52, 365, 365*2, 365*24]  # verticals black line to spot the day, the week and the month
    ylabel = ['Year','Month', 'Week', 'Day', '12h','Hour']
    # --------
    yf_abs = 2.0/N * np.abs(yf[:N//2])
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.set_xscale('log')
    plt.plot(yf_abs)
    plt.xlim(0.9,365*64/2)

    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    plt.xlabel('Time, log scale', fontsize=20, fontweight='bold')
    plt.title('FFT of the signal "'+ signal +'" in ' + year , fontsize=20, fontweight='bold')

    plt.grid(True, which="both")
    # Options of the log grid
#     locmin = mticker.LogLocator(base=10, subs=np.arange(0.1,1,0.1),numticks=10)
#     g.ax.xaxis.set_minor_locator(locmin)
#     g.ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    plt.rc('font', family='serif')
    # -------------- Vertical lines -----------
    for xc in xcoords:
        plt.axvline(x=xc,linewidth=1.2, color='black', linestyle='--', alpha = 0.5)
    plt.xticks(xcoords,ylabel)
    plt.show(block=False)
    # plt.pause(3)
    # plt.close()

def plot_EPN(emax, pmax, n, uf, serv, time_scales, satisfactions, scenario_name):

    #
    # Aesthetic settings
    # Plot aesthetic settings
    sns.set()
    sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("colorblind")  # set colors palettte
    plt.rc('text', usetex=False)

    markers = ['o', 'v', 's', '^', 'o', 'v', 's', '^', 'o', 'v', 's', '^']
    markers = ''.join(markers)
    mark_size = 10

    xcoords = [24, 7 * 24, 30 * 24, 365 * 24]  # verticals black line to spot the day, the week and the month

    #     labels = [None, r'10 \%', None, None, r'90 \%',None, None, None, r'100 \%']
    labels = [str(satis)+' %' for satis in satisfactions]

    # ----- Figure settings

    # plt.rc('text', usetex=True)  # To get Latex style in the figures
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.rc('lines', linewidth=3)

    # Then , one by one

    ##---- create figure ----

    fwidth = 12.  # total width of the figure in inches
    fheight = 8.  # total height of the figure in inches

    fig = plt.figure(figsize=(fwidth, fheight))

    # ---- define margins -> size in inches / figure dimension ----

    left_margin = 0.95 / fwidth
    right_margin = 0.2 / fwidth
    bottom_margin = 0.5 / fheight
    top_margin = 0.25 / fheight

    #     #---- create axes ----

    #     # dimensions are calculated relative to the figure size

    x = left_margin  # horiz. position of bottom-left corner
    y = bottom_margin  # vert. position of bottom-left corner
    w = 1 - (left_margin + right_margin)  # width of axes
    h = 1 - (bottom_margin + top_margin)  # height of axes

    ax = fig.add_axes([x, y, w, h])

    #     #---- Define the Ylabel position ----

    # Location are defined in dimension relative to the figure size

    xloc = 0.25 / fwidth
    yloc = y + h / 2.

    plt.close('all')
    ##
    # Usage factor
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()

    plt.ylabel(r"Utilization factor factor ($\%$)")
    plt.xscale('log')
    plt.xlabel("cycle length (h)")
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])
    plt.grid(True, which="both")
    lines = plt.plot(time_scales, uf)
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()
    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')
    plt.ylim(0, 105)

    plt.title(scenario_name)

    #     plt.savefig(save_directory+data_name +'_uf' +'.'+ extension,  dpi=600, bbox_inches = 'tight')


    ##
    # Energy
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()
    plt.yscale('log')
    plt.ylabel('Energy (MWh)')
    plt.xlabel("cycle length (h)")
    plt.xscale('log')
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])

    plt.grid(True, which="both")
    lines = plt.plot(time_scales, emax)
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()
    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')


    plt.title(scenario_name)
    #     plt.savefig(save_directory+data_name +'_energy' +'.'+ extension,  dpi=600, bbox_inches = 'tight')

    #     ##
    # Service
    plt.figure(figsize=(fwidth, fheight))
    plt.subplot()
    plt.xscale('log')
    plt.xlabel("cycle length (h)")
    plt.ylabel(r"E$\cdot n_{cycles}$ (MWh/year)")
    plt.xticks([0.75, 3, 10, 24, 168, 720, 8760], ['0.75', '3', '10', 'day', 'week', 'month', 'year'])
    lines = plt.plot(time_scales, serv)
    plt.grid(True, which="both")
    for i in range(len(lines)):
        lines[i].set_visible(labels[i] is not None)
        lines[i].set_marker(markers[i])
        lines[i].set_markersize(mark_size)
    for xc in xcoords:
        plt.axvline(x=xc, linewidth=1.2, color='black', linestyle='--', label='time')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    plt.tight_layout()

    plt.legend([lines[i] for i, lab in enumerate(labels) if lab is not None],
               [labels[i] for i, lab in enumerate(labels) if lab is not None], loc='upper left')

    ax.set_ylabel('yLabel', fontsize=16, verticalalignment='top',
                  horizontalalignment='center')
    ax.yaxis.set_label_coords(xloc, yloc, transform=fig.transFigure)
    # plt.ylim(0,3e8)
    plt.title(scenario_name)

    plt.show()