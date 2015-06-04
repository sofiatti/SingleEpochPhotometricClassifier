import argparse
import sncosmo
import gzip
import time
import os
import numpy as np
import cPickle as pickle
from scipy import io, interpolate, optimize
from numpy.random import seed, normal, uniform, choice

parser = argparse.ArgumentParser()
parser.add_argument(
                dest='n', type=int,
                help='Number of MC itirations ex.:1000')
parser.add_argument(
                dest='phase_min', type=int,
                help='Minimum phase ex.: 0')
parser.add_argument(
                dest='phase_max', type=int,
                help='Maximum phase ex.: 5')
parser.add_argument(
                dest='z_min', type=float,
                help='Minimum redshift ex.: 2')
parser.add_argument(
                dest='z_max', type=float,
                help='Maximum redshift ex.: 0.5')
parser.add_argument(
                dest='z_interval', type=float,
                help='Redshift interval ex.: 0.05')
parser.add_argument(
                dest='filters', type=list, nargs='?',
                default=['f105w', 'f140w', 'f160w', 'uvf814w'],
                help='List of strings where the strings are filter names.')
args = parser.parse_args()

my_dir = '/Users/carolinesofiatti/projects/classification/data/'
filters = parser.get_default('filters')

t0 = 0
hostr_v = 3.1
dust = sncosmo.CCM89Dust()
seed(11)

zero_point = {'f105w': 26.235, 'f140w': 26.437, 'f160w': 25.921,
              'uvf814w': 25.0985, 'zpsys': 'ab'}

all_model_Ibc = ['s11-2005hl', 's11-2005hm', 's11-2006fo', 'nugent-sn1bc',
                 'nugent-hyper', 's11-2006jo', 'snana-2004fe',
                 'snana-2004gq', 'snana-sdss004012', 'snana-2006fo',
                 'snana-sdss014475', 'snana-2006lc', 'snana-04d1la',
                 'snana-04d4jv', 'snana-2004gv', 'snana-2006ep',
                 'snana-2007y', 'snana-2004ib', 'snana-2005hm',
                 'snana-2006jo', 'snana-2007nc']


all_model_II = ['s11-2005lc', 's11-2005gi', 's11-2006jl', 'nugent-sn2p',
                'snana-2004hx', 'snana-2005gi', 'snana-2006gq',
                'snana-2006kn', 'snana-2006jl', 'snana-2006iw',
                'snana-2006kv', 'snana-2006ns', 'snana-2007iz',
                'snana-2007nr', 'snana-2007nr', 'snana-2007kw',
                'snana-2007ky', 'snana-2007lj', 'snana-2007lb',
                'snana-2007ll', 'snana-2007nw', 'snana-2007ld',
                'snana-2007md', 'snana-2007lz', 'snana-2007lx',
                'snana-2007og', 'snana-2007ny', 'snana-2007nv',
                'snana-2007pg', 's11-2004hx', 'nugent-sn2l', 'nugent-sn2n',
                'snana-2006ez', 'snana-2006ix']

def all_mabs(sne_type):
    # The majority of Type II are IIp
    my_mabs = {'Ibc': normal(-17.6, scale=1),
               'II': normal(-16.80, scale=0.97),
               'IIn': normal(-18.62, scale=1.48),
               'IIl': normal(-17.98, scale=0.9)}
    return(my_mabs[sne_type])

def which_salt(z):
    salt_name = 'salt2'
    salt_version = '2.4'

    rest_salt_max_wav = 9200
    rest_salt_min_wav = 2000

    salt_max_wav = (1 + z) * rest_salt_max_wav
    salt_min_wav = (1 + z) * rest_salt_min_wav

    for filter in filters:
        band = sncosmo.get_bandpass(filter)
        if (band.wave[0] < salt_min_wav or band.wave[-1] > salt_max_wav):
            salt_name = 'salt2-extended'
            salt_version = '1.0'
            break
    return salt_name, salt_version


def obsflux_Ia(z, my_phase):
    """Given a filter, redshift z at given phase, generates the observed
    magnitude for SNe Type Ia"""

    alpha = 0.12
    beta = 3.
    x1 = normal(0., 1.)
    c = normal(0., 0.1)
    mabs = normal(-19.1 - alpha*x1 + beta*c, scale=0.15)
    zpsys = zero_point['zpsys']
    salt_name, salt_version = which_salt(z)
    model_Ia = sncosmo.Model(source=sncosmo.get_source(salt_name,
                                                       version=salt_version))
    model_Ia.set(z=z)
    model_Ia.set_source_peakabsmag(mabs, 'bessellb', 'vega')
    p = {'z': z, 't0': t0, 'x1': x1, 'c': c}
    p['x0'] = model_Ia.get('x0')
    model_Ia.set(**p)
    phase = my_phase + model_Ia.source.peakphase('bessellb')

    keys = filters
    values = np.zeros(len(filters))
    all_obsflux_Ia = dict(zip(keys, values))

    for filter in filters:
        zp = zero_point[filter]
        obsflux_Ia = model_Ia.bandflux(filter, t0+(phase * (1+z)), zp, zpsys)
        all_obsflux_Ia[filter] = obsflux_Ia
    return all_obsflux_Ia, p, salt_name, salt_version


def obsflux_cc(z, sn_type, all_model):
    """Given a filter and redshift z, generates the observed
    magnitude for SNe Type Ibc or Type II"""
    zpsys = zero_point['zpsys']
    my_model = choice(all_model)
    if my_model in ['s11-2004hx', 'nugent-sn2l']:
        sn_type = 'IIl'
    elif my_model in ['nugent-sn2n', 'snana-2006ez', 'snana-20069ix']:
        sn_type = 'IIn'
    model = sncosmo.Model(source=sncosmo.get_source(my_model),
                          effects=[dust], effect_names=['host'],
                          effect_frames=['rest'])
    mabs = all_mabs(sn_type)
    model.set(z=z)
    model.set_source_peakabsmag(mabs, 'bessellb', 'vega')
    p_core_collapse = {'z': z, 't0': t0, 'hostebv': uniform(-0.1, 0.65),
                       'hostr_v': hostr_v}
    model.set(**p_core_collapse)
    my_phase = uniform(0., 5.)
    phase = my_phase + model.source.peakphase('bessellb')

    keys = filters
    values = np.zeros(len(filters))
    all_obsflux = dict(zip(keys, values))
    for filter in filters:
        zp = zero_point[filter]
        obsflux = model.bandflux(filter, t0+(phase * (1+z)), zp, zpsys)
        all_obsflux[filter] = obsflux
    return all_obsflux, p_core_collapse


def save_file(object, filename, protocol=-1):
    """Saves a compressed object to disk"""
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, file, protocol)
    file.close()
    return ()


def z_from_photo_z(photo_z_file, n, my_z_array=None):
    my_p_z = io.readsav(photo_z_file)
    pz = my_p_z['p_z']
    if my_z_array is None:
        z = np.arange(0, 5, .01)
    else:
        z = my_z_array
    if np.shape(pz) != np.shape(z):
        raise ValueError("p_z array and z array are different sizes")
    dz = z[1] - z[0]
    pz /= (dz * pz).sum()
    ecdf = np.cumsum(pz * dz)
    cdf = interpolate.interp1d(z, ecdf)

    def func(x, *args):
        my_cdf = args[0]
        cdf = args[1]
        return abs(my_cdf - cdf(x))
    out_z = []
    for i in range(n):
        my_cdf = np.random.uniform(0, 1)
        my_z = optimize.fmin(func, (1.5), args=(my_cdf, cdf), disp=0)
        out_z.append(my_z[0])
    out_z = np.asarray(out_z)
    return out_z


def mc_file(n, min_phase, max_phase, z=None, photo_z_file=None,
            z_for_photo_z_file=None):
    """Returns 3 arrays. For each SNe type, returns an array with the observed
    filter flux.

    z is either an array (for photo_z_file) or a number"""

    if z is None:
        z_array = z_from_photo_z(photo_z_file, n, z_for_photo_z_file)

    else:
        z_array = np.ones(n)*z

    files_dir = my_dir + 'mc_files_%.0f_%.0f_%.0f' % (n, min_phase, max_phase)
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    type_Ia_flux = []
    type_Ibc_flux = []
    type_II_flux = []

    p_Ia = []
    p_Ibc = []
    p_II = []

    salt_name = []
    salt_version = []

    for i in range(n):
#        if i % (n/100) == 0:
#            print i/(n/100), '% complete'
        my_phase = uniform(min_phase, max_phase)

        my_obsflux_Ia, my_p_Ia, my_salt_name, my_salt_version = obsflux_Ia(
                        z_array[i], my_phase)
        type_Ia_flux.append(my_obsflux_Ia)
        p_Ia.append(my_p_Ia)
        salt_name.append(my_salt_name)
        salt_version.append(my_salt_version)

        my_obsflux_Ibc, my_p_Ibc = obsflux_cc(z_array[i], 'Ibc',
                                              all_model_Ibc)
        type_Ibc_flux. append(my_obsflux_Ibc)
        p_Ibc.append(my_p_Ibc)

        my_obsflux_II, my_p_II = obsflux_cc(z_array[i], 'II',
                                            all_model_II)
        type_II_flux.append(my_obsflux_II)
        p_II.append(my_p_II)

    if z is None:
        new_fname = files_dir + '/simulated_mc.gz'
        keys = ['type_Ia_flux', 'type_Ia_params', 'type_Ibc_flux',
                'type_Ibc_params', 'type_II_flux', 'type_II_params', 'z',
                'filter', 'salt_name', 'salt_version']
        values = [type_Ia_flux, p_Ia, type_Ibc_flux, p_Ibc, type_II_flux, p_II,
                  z_array, filter, salt_name, salt_version]

    else:
        fname = my_dir + '/test/check/z%0.2f' % z_array[i] + '_simulated_mc.gz'
        # now we want to remove the dot
        i = fname.index('.')
        new_fname = fname[:i] + fname[i+1:]
        keys = ['type_Ia_flux', 'type_Ia_params', 'type_Ibc_flux',
                'type_Ibc_params', 'type_II_flux', 'type_II_params', 'filter',
                'salt_name', 'salt_version']
        values = [type_Ia_flux, p_Ia, type_Ibc_flux, p_Ibc, type_II_flux, p_II,
                  filter, salt_name, salt_version]
    mc = dict(zip(keys, values))
    # pickle.dump( montecarlo, open( new_fname, 'wb'))
    save_file(mc, new_fname)
    return()

all_z = np.arange(args.z_min, args.z_max + args.z_interval, args.z_interval)
for i in all_z:
    start = time.time()
    mc_file(args.n, args.phase_min, args.phase_max, i)
    print 'It took', time.time() - start, 'seconds.'

'''
Think about this one some more:

start = time.time()
#mc_file(args.n, 'f105w', args.phase_min, args.phase_max,
         photo_z_file='zprob_v0.06_09.000129')
mc_file(args.n, 'f105w', 0., 5., z=1.23)
print 'It took', time.time() - start, 'seconds.'
'''
