import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from funcsForSimulatedData import load, file_name, add_error


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def kde3d(x, y, z, delta, data_point):

    values = np.vstack([x, y, z]).T
    # Create a regular 3D grid with 50 points in each dimension
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()

    xem = np.arange(xmin, xmax + delta, delta)
    yem = np.arange(ymin, ymax + delta, delta)
    zem = np.arange(zmin, zmax + delta, delta)

    xi, yi, zi = np.meshgrid(xem, yem, zem, indexing='ij')
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    kde_coords = coords.T
    # Evaluate the KDE on a regular grid.
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(values)
    kde = grid.best_estimator_
    log_pdf = kde.score_samples(kde_coords)
    pdf = np.exp(log_pdf)
    pdf = pdf.reshape(len(xem), len(yem), len(zem))

    x_bin, x_idx = find_nearest(xem, data_point[0])
    y_bin, y_idx = find_nearest(yem, data_point[1])
    z_bin, z_idx = find_nearest(zem, data_point[2])

    data_H = pdf[x_idx][y_idx][z_idx]
    H = np.sort(pdf.ravel())[::-1]
    percentile = sum(H[H > data_H])/H.sum()
    return (percentile)


def find_percentile(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
                    flux_filter2, flux_filter3, flux_filter1_err,
                    flux_filter2_err, flux_filter3_err, z):

    all_percentile = []
    data_point = (flux_filter1, flux_filter2, flux_filter3)
    data_point = np.asarray(data_point)

    mc_file_name = file_name(my_dir, file_dir, z)
    mc_dict = load(mc_file_name)
    mc_dict = add_error(mc_dict, filter1, flux_filter1_err)
    mc_dict = add_error(mc_dict, filter2, flux_filter2_err)
    mc_dict = add_error(mc_dict, filter3, flux_filter3_err)

    Ia_percentile = kde3d(mc_dict['type_Ia']['flux'][filter1],
                          mc_dict['type_Ia']['flux'][filter2],
                          mc_dict['type_Ia']['flux'][filter3], .3, data_point)

    Ibc_percentile = kde3d(mc_dict['type_Ibc']['flux'][filter1],
                           mc_dict['type_Ibc']['flux'][filter2],
                           mc_dict['type_Ibc']['flux'][filter3], .3, data_point)

    II_percentile = kde3d(mc_dict['type_II']['flux'][filter1],
                          mc_dict['type_II']['flux'][filter2],
                          mc_dict['type_II']['flux'][filter3], .3, data_point)

    my_percentile = (Ia_percentile, Ibc_percentile, II_percentile)
    my_percentile = np.asarray(my_percentile)
    all_percentile.append(my_percentile)
    all_percentile = np.squeeze(all_percentile)

    survival_f = 1 - all_percentile
    return (survival_f)
