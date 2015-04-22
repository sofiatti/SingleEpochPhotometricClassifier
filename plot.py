import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numpy as np
import time
import math
from scipy.stats import norm
from MCtoFlux import get_fluxes
from matplotlib import rc
from zIter import iterator

t0 = time.time()
my_dir = "/home/sofiatti/projects/scp/"
file_dir = "mc_files_1000_-5_0/"

filter1 = 'f105w'
flux_filter1 = 1.89
flux_filter1_err = 0.19

filter2 = 'f140w'
flux_filter2 = 1.96 
flux_filter2_err = 0.23

filter3 = 'uvf814w'
flux_filter3 = 0.6
flux_filter3_err = 0.18

def plot(x, y, title, xlabel, ylabel, file_name):
    sns.set_context("talk", font_scale=1.4)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    print x
    print y

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    plt.savefig(file_name)
    plt.close()
'''
def plotScatter():
    type_Ia_flux_filter1, type_Ia_flux_filter2, type_Ia_flux_filter3, type_Ibc_flux_filter1, type_Ibc_flux_filter2, type_Ibc_flux_filter3, type_II_flux_filter1, type_II_flux_filter2, type_II_flux_filter3 = get_fluxes(my_dir, file_dir, filter1, filter2, filter3, z)

    
def plotContour():
    type_Ia_flux_filter1, type_Ia_flux_filter2, type_Ia_flux_filter3, type_Ibc_flux_filter1, type_Ibc_flux_filter2, type_Ibc_flux_filter3, type_II_flux_filter1, type_II_flux_filter2, type_II_flux_filter3 = get_fluxes(my_dir, file_dir, filter1, filter2, filter3, z)
'''

def plotScore(type=None, file=None, file_z = None, mean=None, sigma=None):
    score, my_z = iterator(my_dir, file_dir, filter1, filter2, filter3, flux_filter1, flux_filter2, flux_filter3, type='score')
    if type = 'pz':
	pz_file = io.readsav(file)
        pz = pz_file['p_z']
        if file_z is None:
            z = np.arange(0, 5, .01)
        else:
            z = my_z
        if np.shape(pz) != np.shape(z):
            print np.shape(pz), np.shape(z)
            raise ValueError("pz array and z array are different sizes!")
        pdf = interpolate.interp1d(z, pz)
	pdf = pdf/pdf.sum()
	product = np.multiply(score.T, pdf)
	plot(my_z, product, 'SF x Photo-z', 'z', '', 'SFxPhoto_z.png')
    elif type = 'gauss':
        pdf = mlab.normpdf(my_z,mean,sigma)
	product = np.muliply(score.T, pdf)
	plot(my_z, product, 'SF x Gaussian PDF', 'z', '', 'SFxGaussianPDF.png')
    
    else:
	plot(my_z, score, 'Survival Function', 'z', '(1 - percentile)', 'SF.png')
	 
def plotPDF():
    pdf, my_z = iterator(my_dir, file_dir, filter1, filter2, filter3, flux_filter1, flux_filter2, flux_filter3, type='pdf')
    plot(my_z, pdf, 'Random Forest: All Filters', 'z', 'PDF', 'RFC_PDF.png')

def plotPDFxScore():
    pdf, my_z = iterator(my_dir, file_dir, filter1, filter2, filter3, flux_filter1, flux_filter2, flux_filter3, type='pdf')

    score, my_z = iterator(my_dir, file_dir, filter1, filter2, filter3, flux_filter1, flux_filter2, flux_filter3, type='score')

    pdf = np.asarray(pdf)
    score = np.asarray(score)
    product = np.multiply(pdf, score)
    
    plot(my_z, product, 'Random Forest x Score', 'z', 'PDF x Score', 'RFCxScore.png')
