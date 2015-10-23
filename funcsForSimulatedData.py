import gzip
import cPickle
import numpy as np


def load(file_name):
    """Loads a compressed object from disk"""
    file = gzip.GzipFile(file_name, 'rb')
    object = cPickle.load(file)
    file.close()
    return object


def save(object, file_name, protocol=-1):
    """Saves a compressed object to disk"""
    file = gzip.GzipFile(file_name, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    return ()


def mask(my_flux, percent):
    """Removes outliers. Only keeps data within input percentile"""
    outlier_mask = my_flux < np.percentile(my_flux, percent)
    flux = my_flux[outlier_mask]
    return flux


def add_error(mc_dict, filter, flux_filter_err):
    for sne_type in mc_dict.keys():
        mc_dict[sne_type]['flux'][filter] += np.random.normal(
         loc=0, scale=flux_filter_err,
         size=np.shape(mc_dict[sne_type]['flux'][filter]))
    return mc_dict


def dict_from_list(list_of_dicts):
    ds = list_of_dicts
    d = {}
    for k in ds[0].iterkeys():
        d[k] = tuple(d[k] for d in ds)
    return d


def file_name(my_dir, file_dir, z):
    z = z * 100
    file_z = '%.0f' % z
    if len(file_z) < 3:
        file_z = '0' + file_z
    name = my_dir + file_dir + 'z' + file_z + '_' + 'simulated_mc.gz'
    return name
