import glob
import pandas as pd
import copy
import gzip
import cPickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load(filename):
    """Loads a compressed object from disk"""
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
    return object


def make_csv(my_dir, file_dir, filter1, filter2, filter3, z):
    file_z = str(z)
    if len(file_z) < 3:
        file_z = '0' + file_z
    dict_filter1 = load(my_dir + file_dir + 'z' + file_z + '_' + filter1 +
                        '_mc.gz')
    dict_filter2 = load(my_dir + file_dir + 'z' + file_z + '_' + filter2 +
                        '_mc.gz')
    dict_filter3 = load(my_dir + file_dir + 'z' + file_z + '_' + filter3 +
                        '_mc.gz')

    type_Ia_flux_filter1 = dict_filter1['type_Ia_flux']
    type_Ibc_flux_filter1 = dict_filter1['type_Ibc_flux']
    type_II_flux_filter1 = dict_filter1['type_II_flux']

    type_Ia_flux_filter2 = dict_filter2['type_Ia_flux']
    type_Ibc_flux_filter2 = dict_filter2['type_Ibc_flux']
    type_II_flux_filter2 = dict_filter2['type_II_flux']

    type_Ia_flux_filter3 = dict_filter3['type_Ia_flux']
    type_Ibc_flux_filter3 = dict_filter3['type_Ibc_flux']
    type_II_flux_filter3 = dict_filter3['type_II_flux']

    flux_filter1 = np.hstack((type_Ia_flux_filter1, type_Ibc_flux_filter1,
                             type_II_flux_filter1))

    flux_filter2 = np.hstack((type_Ia_flux_filter2, type_Ibc_flux_filter2,
                              type_II_flux_filter2))

    flux_filter3 = np.hstack((type_Ia_flux_filter3, type_Ibc_flux_filter3,
                             type_II_flux_filter3))

    np.savetxt('new_data.csv', np.transpose([flux_filter1, flux_filter2,
               flux_filter3]), delimiter=',', header='Flux filter1, '
               'Flux filter2, Flux filter3')

# Read in data generated using sncosmo and assign the correct SN Type to each
# instance


def read_data(files_dir, filename):
    indices = [i for i, x in enumerate(files_dir) if x == '_']
    n = files_dir[(indices[1] + 1):indices[2]]
    n_Ibc_start = int(n)
    n_II_start = 2 * n_Ibc_start

    all_sources = pd.read_csv(filename)
    all_sources['Type'] = 'Type Ia'
    all_sources.ix[n_Ibc_start:n_II_start, 'Type'] = 'Type Ibc'
    all_sources.ix[n_II_start:, 'Type'] = 'Type II'
    all_features = copy.copy(all_sources)
    all_label = all_sources["Type"]
    del all_features["Type"]

    X = copy.copy(all_features.values)
    Y = copy.copy(all_label.values)

    return X, Y


def obtain_proba(files_dir, flux_filter1, flux_filter2, flux_filter3):

    X, Y = read_data(files_dir, 'new_data.csv')

    Y[Y == "Type Ia"] = 0
    Y[Y == "Type Ibc"] = 1
    Y[Y == "Type II"] = 2
    Y = Y.astype(int)

    clf = RandomForestClassifier(n_estimators=200, oob_score=True)
    rfc = clf.fit(X, Y)
    proba = rfc.predict_proba([flux_filter1, flux_filter2, flux_filter3])
    return(proba)

