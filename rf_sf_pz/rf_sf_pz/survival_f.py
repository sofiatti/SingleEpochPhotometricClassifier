import numpy as np
from scipy import stats
from load import load, mask


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def kde3d(x, y, z, data_point):

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)

    values = np.vstack([x, y, z])
    kde = stats.gaussian_kde(values)

    # Create a regular 3D grid with 50 points in each dimension
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]

    # Evaluate the KDE on a regular grid...
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    kde_grid = density
    xem = coords[0].reshape(50, 2500)[:, 0]
    yem = coords[1].reshape(2500, 50)[:, 0]
    zem = coords[2].reshape(50, 2500)[0, :]

    x_bin, x_idx = find_nearest(xem, data_point[0])
    y_bin, y_idx = find_nearest(yem, data_point[1])
    z_bin, z_idx = find_nearest(zem, data_point[2])

    data_H = kde_grid[x_idx][y_idx][z_idx]
    H = np.sort(kde_grid.ravel())[::-1]
    percentile = sum(H[H > data_H])/H.sum()
    return (percentile)


def find_percentile(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
                    flux_filter2, flux_filter3, flux_filter1_err,
                    flux_filter2_err, flux_filter3_err, z):

    all_percentile = []
    data_point = (flux_filter1, flux_filter2, flux_filter3)
    data_point = np.asarray(data_point)

    z = z * 100
    file_z = '%.0f' % z
    if len(file_z) < 3:
        file_z = '0' + file_z
    dict_filter1 = load(my_dir + file_dir + 'z' + file_z + '_' + filter1 +
                        '_mc.gz')
    dict_filter2 = load(my_dir + file_dir + 'z' + file_z + '_' + filter2 +
                        '_mc.gz')
    dict_filter3 = load(my_dir + file_dir + 'z' + file_z + '_' + filter3 +
                        '_mc.gz')

    dict_filter1['type_Ia_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter1_err,
                                size=np.shape(dict_filter1['type_Ia_flux']))
    type_Ia_flux_filter1 = mask(dict_filter1['type_Ia_flux'])
    dict_filter1['type_Ibc_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter1_err,
                                size=np.shape(dict_filter1['type_Ibc_flux']))
    type_Ibc_flux_filter1 = mask(dict_filter1['type_Ibc_flux'])
    dict_filter1['type_II_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter1_err,
                                size=np.shape(dict_filter1['type_II_flux']))
    type_II_flux_filter1 = mask(dict_filter1['type_II_flux'])

    dict_filter2['type_Ia_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter2_err,
                                size=np.shape(dict_filter2['type_Ia_flux']))
    type_Ia_flux_filter2 = mask(dict_filter2['type_Ia_flux'])
    dict_filter2['type_Ibc_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter2_err,
                                size=np.shape(dict_filter2['type_Ibc_flux']))
    type_Ibc_flux_filter2 = mask(dict_filter2['type_Ibc_flux'])
    dict_filter2['type_II_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter2_err,
                                size=np.shape(dict_filter2['type_II_flux']))
    type_II_flux_filter2 = mask(dict_filter2['type_II_flux'])

    dict_filter3['type_Ia_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter3_err,
                                size=np.shape(dict_filter3['type_Ia_flux']))
    type_Ia_flux_filter3 = mask(dict_filter3['type_Ia_flux'])
    dict_filter3['type_Ibc_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter3_err,
                                size=np.shape(dict_filter3['type_Ibc_flux']))
    type_Ibc_flux_filter3 = mask(dict_filter3['type_Ibc_flux'])
    dict_filter3['type_II_flux'] += np.random.normal(
                                loc=0,
                                scale=flux_filter3_err,
                                size=np.shape(dict_filter3['type_II_flux']))
    type_II_flux_filter3 = mask(dict_filter3['type_II_flux'])

    Ia_percentile = kde3d(type_Ia_flux_filter1, type_Ia_flux_filter2,
                          type_Ia_flux_filter3, data_point)
    Ibc_percentile = kde3d(type_Ibc_flux_filter1, type_Ibc_flux_filter2,
                           type_Ibc_flux_filter3, data_point)
    II_percentile = kde3d(type_II_flux_filter1, type_II_flux_filter2,
                          type_II_flux_filter3, data_point)

    my_percentile = (Ia_percentile, Ibc_percentile, II_percentile)
    my_percentile = np.asarray(my_percentile)
    all_percentile.append(my_percentile)
    all_percentile = np.squeeze(all_percentile)

    survival_f = 1 - all_percentile
    return (survival_f)
