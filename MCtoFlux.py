import numpy as np
from load import load, mask




def get_fluxes(my_dir, file_dir, filter1, filter2, filter3, z):
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
    type_Ia_flux_filter1 = mask(type_Ia_flux_filter1)

    type_Ibc_flux_filter1 = dict_filter1['type_Ibc_flux']
    type_Ibc_flux_filter1 = mask(type_Ibc_flux_filter1)

    type_II_flux_filter1 = dict_filter1['type_II_flux']
    type_II_flux_filter1 = mask(type_II_flux_filter1)

    type_Ia_flux_filter2 = dict_filter2['type_Ia_flux']
    type_Ia_flux_filter2 = mask(type_Ia_flux_filter2)

    type_Ibc_flux_filter2 = dict_filter2['type_Ibc_flux']
    type_Ibc_flux_filter2 = mask(type_Ibc_flux_filter2)

    type_II_flux_filter2 = dict_filter2['type_II_flux']
    type_II_flux_filter2 = mask(type_II_flux_filter2)

    type_Ia_flux_filter3 = dict_filter3['type_Ia_flux']
    type_Ia_flux_filter3 = mask(type_Ia_flux_filter3)

    type_Ibc_flux_filter3 = dict_filter3['type_Ibc_flux']
    type_Ibc_flux_filter3 = mask(type_Ibc_flux_filter3)

    type_II_flux_filter3 = dict_filter3['type_II_flux']
    type_II_flux_filter3 = mask(type_II_flux_filter3)

    return(type_Ia_flux_filter1, type_Ia_flux_filter2, type_Ia_flux_filter3,
           type_Ibc_flux_filter1, type_Ibc_flux_filter2, type_Ibc_flux_filter3,
           type_II_flux_filter1, type_II_flux_filter2, type_II_flux_filter3)

