import gzip
import cPickle
import numpy as np


def save_file(object, filename, protocol=-1):
    """Saves a compressed object to disk"""
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    return ()

def load(filename):
    """Loads a compressed object from disk"""
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
    return object


def mask(my_flux):
    """Removes outliers. Only keeps data within 95th percentile"""
    outlier_mask = my_flux < np.percentile(my_flux, 95)
    flux = my_flux[outlier_mask]
    return flux


def get_dict(list_of_dicts):
    ds = list_of_dicts
    d = {}
    for k in ds[0].iterkeys():
        d[k] = tuple(d[k] for d in ds)
    return d
