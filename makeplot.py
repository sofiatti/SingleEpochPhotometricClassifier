import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from load import load
from zIter import iterator


def file_name(my_dir, file_dir, filter, z):
    z = z * 100
    file_z = '%.0f' % z
    if len(file_z) < 3:
        file_z = '0' + file_z
    name = my_dir + file_dir + 'z' + file_z + '_' + filter + '_mc.gz'
    return name


def contour(my_dir, file_dir, filter1, filter2, z, point_flux_filter1,
            point_flux_filter1_err, point_flux_filter2,
            point_flux_filter2_err):
    """Generates a plot and transforms the figure into an HTML file from the
    montecarlo data. Adds a point to the data"""
    point_flux_diff = point_flux_filter2 - point_flux_filter1
    point_flux_diff_err = point_flux_filter2_err + point_flux_filter1_err

    dict_filter1 = load(file_name(my_dir, file_dir, filter1, z))
    dict_filter2 = load(file_name(my_dir, file_dir, filter2, z))

    type_Ia_flux_filter1 = dict_filter1['type_Ia_flux']
    type_Ibc_flux_filter1 = dict_filter1['type_Ibc_flux']
    type_II_flux_filter1 = dict_filter1['type_II_flux']
    filter1 = dict_filter1['filter']

    type_Ia_flux_filter2 = dict_filter2['type_Ia_flux']
    type_Ibc_flux_filter2 = dict_filter2['type_Ibc_flux']
    type_II_flux_filter2 = dict_filter2['type_II_flux']
    filter2 = dict_filter2['filter']

    type_Ia_flux_diff = np.subtract(type_Ia_flux_filter2, type_Ia_flux_filter1)
    type_Ibc_flux_diff = np.subtract(type_Ibc_flux_filter2,
                                     type_Ibc_flux_filter1)
    type_II_flux_diff = np.subtract(type_II_flux_filter2, type_II_flux_filter1)

    flux = [type_Ia_flux_filter1, type_Ibc_flux_filter1, type_II_flux_filter1]
    diff = [type_Ia_flux_diff, type_Ibc_flux_diff, type_II_flux_diff]

    for i, item in enumerate(flux):
        outlier_mask = (flux[i] < np.percentile(
                        flux[i], 95)) & (diff[i] < np.percentile(diff[i], 95))
        flux[i] = flux[i][outlier_mask]
        diff[i] = diff[i][outlier_mask]

    sns.set(style="white", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    g = sns.JointGrid(filter2.upper() + " - " + filter1.upper() +
                      " flux difference", filter1.upper() + " flux", space=0)

    sns.kdeplot(diff[0], ax=g.ax_marg_x, legend=True, label='Type Ia')
    sns.kdeplot(flux[0], ax=g.ax_marg_y, vertical=True, legend=False)
    sns.kdeplot(diff[0], flux[0], cut=4, ax=g.ax_joint,
                alpha=0.8, cmap='Blues_d')

    sns.kdeplot(diff[1], ax=g.ax_marg_x, legend=True, label='Type Ib/c')
    sns.kdeplot(flux[1], ax=g.ax_marg_y, vertical=True, legend=False)
    sns.kdeplot(diff[1], flux[1], cut=4, ax=g.ax_joint,
                alpha=0.8, cmap='Greens_d')

    sns.kdeplot(diff[2], ax=g.ax_marg_x, legend=True, label='Type II')
    sns.kdeplot(flux[2], ax=g.ax_marg_y, vertical=True, legend=False)
    sns.kdeplot(diff[2], flux[2], cut=4, ax=g.ax_joint,
                alpha=0.8, cmap='Reds_d')

    g.ax_joint.errorbar(point_flux_diff, point_flux_filter1,
                        xerr=point_flux_diff_err, yerr=point_flux_filter1_err,
                        fmt='--o', c='k')
    g.set_axis_labels(xlabel=filter2.upper() + " - " + filter1.upper() +
                      " flux difference (counts/s)",
                      ylabel=filter1.upper() + " flux (counts/s)")
    plt.tight_layout()
    plt.show()
    '''
    # plt.savefig(filename_filter1[:-5] + filename_filter2[5:-5]+ 'plot.png')
    # mpld3.save_html(plt.gcf(), filename_filter1[:-5] +
                      filename_filter2[5:-5] + 'plot.html')
    '''


def scatter(my_dir, file_dir, filter1, filter2, z):
    sns.set(style="white", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    dict_filter1 = load(file_name(my_dir, file_dir, filter1, z))
    dict_filter2 = load(file_name(my_dir, file_dir, filter2, z))

    type_Ia_flux_filter1 = dict_filter1['type_Ia_flux']
    type_Ibc_flux_filter1 = dict_filter1['type_Ibc_flux']
    type_II_flux_filter1 = dict_filter1['type_II_flux']
    filter1 = dict_filter1['filter']

    type_Ia_flux_filter2 = dict_filter2['type_Ia_flux']
    type_Ibc_flux_filter2 = dict_filter2['type_Ibc_flux']
    type_II_flux_filter2 = dict_filter2['type_II_flux']
    filter2 = dict_filter2['filter']

    type_Ia_flux_diff = np.subtract(type_Ia_flux_filter2, type_Ia_flux_filter1)
    type_Ibc_flux_diff = np.subtract(type_Ibc_flux_filter2,
                                     type_Ibc_flux_filter1)
    type_II_flux_diff = np.subtract(type_II_flux_filter2, type_II_flux_filter1)

    flux = [type_Ia_flux_filter1, type_Ibc_flux_filter1, type_II_flux_filter1]
    diff = [type_Ia_flux_diff, type_Ibc_flux_diff, type_II_flux_diff]

    for i, item in enumerate(flux):
        outlier_mask = (flux[i] < np.percentile(
                        flux[i], 95)) & (diff[i] < np.percentile(diff[i], 95))
        flux[i] = flux[i][outlier_mask]
        diff[i] = diff[i][outlier_mask]

    plt.scatter(diff[0], flux[0], c='b',
                label='Type Ia')
    plt.scatter(diff[1], flux[1], c='g',
                label='Type Ib/c')
    plt.scatter(diff[2], flux[2], c='r',
                label='Type II')
    plt.title('SNCosmo Simulated SNe: z=%.2f' % z)
    plt.xlabel(filter2.upper() + " - " + filter1.upper() +
               " flux difference (counts/s)")
    plt.ylabel(filter1.upper() + " flux (counts/s)")
    plt.legend()
    plt.show()
    plt.close()


def survival(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
             flux_filter2, flux_filter3):
    survival, z = iterator('survival', my_dir, file_dir, filter1, filter2,
                           filter3, flux_filter1, flux_filter2, flux_filter3)

    sns.set(style="darkgrid", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.plot(z, survival)
    plt.title('Survival Function')
    plt.xlabel('z')
    plt.ylabel('1 - CDF')
    plt.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    plt.show()
    # plt.savefig(my_dir + files_dir + file_name)
    plt.close()
    return survival, z


def photoz(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
           flux_filter2, flux_filter3, photo_z_type,
           photo_z_file=None, photo_z_redshift_file=None, mu=None,
           sigma=None):
    photoz, z = iterator('photoz', my_dir, file_dir, filter1, filter2, filter3,
                         flux_filter1, flux_filter2, flux_filter3, photo_z_type,
                         photo_z_file, photo_z_redshift_file, mu, sigma)

    sns.set(style="darkgrid", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.plot(z, photoz)
    plt.xlabel('z')
    plt.ylabel('PDF')
    if photo_z_type == 'file':
        plt.title('Photo-z: ' + photo_z_file)
    else:
        plt.title('Photo-z: Gaussian( mu=%.2f, sigma=%.2f)' % (mu, sigma))
    plt.show()
    # plt.savefig(my_dir + files_dir + file_name)
    plt.close()
    return photoz, z


def random_forest(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
                  flux_filter2, flux_filter3):
    rf, z = iterator('RF', my_dir, file_dir, filter1, filter2, filter3,
                     flux_filter1, flux_filter2, flux_filter3)

    sns.set(style="darkgrid", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.plot(z, rf)
    plt.title('Random Forest')
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    plt.show()
    # plt.savefig(my_dir + files_dir + file_name)
    plt.close()
    return rf, z


def combined(final_pdf, my_dir, file_dir, filter1, filter2, filter3,
             flux_filter1, flux_filter2, flux_filter3, photo_z_type,
             photo_z_file=None, photo_z_redshift_file=None, mu=None,
             sigma=None):
    if final_pdf == 'RF+SF+photoz':
        title = 'Random Forest & Survival Function & Photo-z'
        save_name = 'random_sf_photoz'

        rf, z = random_forest(my_dir, file_dir, filter1, filter2, filter3,
                              flux_filter1, flux_filter2, flux_filter3)
        rf = np.asarray(rf)

        sf, z = survival(my_dir, file_dir, filter1, filter2, filter3,
                         flux_filter1, flux_filter2, flux_filter3)
        sf = np.asarray(sf)

        photo_z, z = photoz(my_dir, file_dir, filter1, filter2, filter3,
                            flux_filter1, flux_filter2, flux_filter3,
                            photo_z_type, photo_z_file, photo_z_redshift_file,
                            mu, sigma)

        first_product = np.multiply(sf, rf)
        product = (np.multiply(first_product.T, photo_z)).T
        # norm_product = product/product.sum()

    elif final_pdf == 'RF+SF':
        title = 'Random Forest & Survival Function'
        save_name = 'random_sf'

        rf, z = random_forest(my_dir, file_dir, filter1, filter2, filter3,
                              flux_filter1, flux_filter2, flux_filter3)
        rf = np.asarray(rf)

        sf, z = survival(my_dir, file_dir, filter1, filter2, filter3,
                         flux_filter1, flux_filter2, flux_filter3)
        sf = np.asarray(sf)

        product = np.multiply(rf, sf)
        product = product.T
        # norm_product = product/product.sum()

    elif final_pdf == 'RF+photoz':
        title = 'Random Forest & Photo-z'
        save_name = 'random_photoz'

        rf, z = random_forest(my_dir, file_dir, filter1, filter2, filter3,
                              flux_filter1, flux_filter2, flux_filter3)
        rf = np.asarray(rf)

        photo_z, z = photoz(my_dir, file_dir, filter1, filter2, filter3,
                            flux_filter1, flux_filter2, flux_filter3,
                            photo_z_type, photo_z_file, photo_z_redshift_file,
                            mu, sigma)

        product = np.multiply(rf.T, photo_z)
        product = product.T
        # norm_product = product/product.sum()

    elif final_pdf == 'SF+photoz':
        title = 'Survival Function & Photo-z'
        save_name = 'sf_photoz'

        sf, z = survival(my_dir, file_dir, filter1, filter2, filter3,
                         flux_filter1, flux_filter2, flux_filter3)
        sf = np.asarray(sf)

        photo_z, z = photoz(my_dir, file_dir, filter1, filter2, filter3,
                            flux_filter1, flux_filter2, flux_filter3,
                            photo_z_type, photo_z_file, photo_z_redshift_file,
                            mu, sigma)
        
        product = np.multiply(sf.T, photo_z)
        product = product.T
        # norm_product = product/product.sum()

    elif final_pdf == 'RF':
        rf, z = random_forest(my_dir, file_dir, filter1, filter2, filter3,
                              flux_filter1, flux_filter2, flux_filter3)
        sys.exit(0)

    elif final_pdf == 'SF':
        sf, z = survival(my_dir, file_dir, filter1, filter2, filter3,
                         flux_filter1, flux_filter2, flux_filter3)
        sys.exit(0)

    elif final_pdf == 'photoz':
        photo_z, z = photoz(my_dir, file_dir, filter1, filter2, filter3,
                            flux_filter1, flux_filter2, flux_filter3,
                            photo_z_type, photo_z_file, photo_z_redshift_file,
                            mu, sigma)
        sys.exit(0)

    sns.set(style="darkgrid", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.plot(z, product)
    plt.title(title)
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    plt.show()
    # plt.savefig(my_dir + files_dir + file_name)
    plt.close()
