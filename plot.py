import sys
import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from zIter import save_arrays, iterator
from load import load


def plot(x, y, ylabel, title, outdir, outname=None):
    sns.set(style="darkgrid", palette="muted")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plot_name = title.replace(' ', '-')
    plot_name = plot_name.lower()
    if outname is not None:
        plot_name = outname

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('z')
    plt.ylabel(ylabel)
    plt.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    plt.savefig(outdir + plot_name + '.png')
    plt.close()


def subplot(x, y, title, outdir, title_photoz=None):
    ''' x, y and title are arrays
    '''
    # Getting a title for the final PDF composed from the other names
    if len(title) == 3:
        final_title = title[0] + ' & ' + title[1] + ' & ' + title[2]
    elif len(title) == 2:
        final_title = title[0] + ' & ' + title[1]
    else:
        final_title = title[0]
    # Getting a plot name for the final PDF composed from the other names
    plot_name = ''
    i = 0
    while i < len(title):
        plot_name += title[i]
        if i < (sum(1 for x in title if x) - 1):
            if title[i]:
                print 'title[i] is' + title[i]
                plot_name += '_'
        i += 1
    plot_name = plot_name.replace(' ', '-')
    plot_name = plot_name.lower()

    if title_photoz is not None:
        title[2] = title_photoz

    sns.set(style="darkgrid", palette="muted")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                               figsize=(12, 12))
    ax1.plot(x[0], y[0])
    ax1.set_title(title[0])
    ax1.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    ax1.set_ylabel('PDF')

    ax2.plot(x[1], y[1])
    ax2.set_title(title[1])
    ax2.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')

    ax3.plot(x[2], y[2])
    ax3.set_title(title[2])
    ax3.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    ax3.set_xlabel('z')
    ax3.set_ylabel('PDF')

    ax4.plot(x[3], y[3])
    ax4.set_title(final_title)
    ax4.legend(['Type Ia', 'Type Ib/c', 'Type II'], loc='upper right')
    ax4.set_xlabel('z')
    plt.savefig(outdir + plot_name + '.png')
    plt.close()


def combined(final_pdf, my_dir, file_dir, filter1, filter2,
             filter3, flux_filter1, flux_filter2, flux_filter3, outdir,
             arrays_file, photo_z_type=None, photo_z_file=None,
             photo_z_redshift_file=None, mu=None, sigma=None):

    if not os.path.isfile(outdir + arrays_file):
        save_arrays(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
                    flux_filter2, flux_filter3, outdir, photo_z_type,
                    photo_z_file, photo_z_redshift_file, mu, sigma)

    my_dict = load(outdir + arrays_file)
    rf = my_dict['rf']
    sf = my_dict['sf']
    z = my_dict['rf_z']

    rf = np.asarray(rf)
    sf = np.asarray(sf)

    if photo_z_type is not None:
        photo_z, my_z = iterator('photo_z', my_dir, file_dir, filter1,
                                 filter2, filter3, flux_filter1, flux_filter2,
                                 flux_filter3, photo_z_type, photo_z_file,
                                 photo_z_redshift_file, mu, sigma)
        if photo_z_type == 'file':
            title_photoz = 'Photo-z: ' + photo_z_file
        else:
            title_photoz = ("Photo-z: Gaussian( mu=%.2f, sigma=%.2f)"
                            % (mu, sigma))

    plt.clf()

    if final_pdf == 'RF+SF+photoz':
        title = ['Random Forest', 'Survival Function', 'Photo-z']
        plot(z, rf, 'PDF', 'Random Forest', outdir)
        plot(z, sf, '1 - CDF', 'Survival Function', outdir)
        plot(z, photo_z, 'PDF', title_photoz, outdir, outname='photoz')

        first_product = np.multiply(sf, rf)
        product = (np.multiply(first_product.T, photo_z)).T

        x = [z, z, z, z]
        y = [rf, sf, photo_z, product]
        subplot(x, y, title, outdir, title_photoz)
        # norm_product = product/product.sum()

    elif final_pdf == 'RF+SF':
        title = ['Random Forest', 'Survival Function', '']
        plot(z, rf, 'PDF', 'Random Forest', outdir)
        plot(z, sf, '1 - CDF', 'Survival Function', outdir)
        product = np.multiply(rf, sf)

        x = [z, z, [], z]
        y = [rf, sf, [], product]
        subplot(x, y, title, outdir)

        # norm_product = product/product.sum()

    elif final_pdf == 'RF+photoz':
        title = ['Random Forest', '', 'Photo-z']
        plot(z, rf, 'PDF', 'Random Forest', outdir)
        plot(z, photo_z, 'PDF', title_photoz, outdir, outname='photoz')

        product = np.multiply(rf.T, photo_z)
        product = product.T
        # norm_product = product/product.sum()

        x = [z, [], z, z]
        y = [rf, [], photo_z, product]
        subplot(x, y, title, outdir, title_photoz)

    elif final_pdf == 'SF+photoz':
        title = ['', 'Survival Function', 'Photo-z']
        plot(z, sf, '1 - CDF', 'Survival Function', outdir)
        plot(z, photo_z, 'PDF', title_photoz, outdir, outname='photoz')

        product = np.multiply(sf.T, photo_z)
        product = product.T
        # norm_product = product/product.sum()

        x = [[], z, z, z]
        y = [[], sf, photo_z, product]
        subplot(x, y, title, outdir, title_photoz)

    elif final_pdf == 'RF':
        plot(z, rf, 'PDF', 'Random Forest', outdir)
        sys.exit(0)

    elif final_pdf == 'SF':
        plot(z, sf, '1 - CDF', 'Survival Function', outdir)
        sys.exit(0)

    elif final_pdf == 'photoz':
        plot(z, photo_z, 'PDF', title_photoz, outdir, outname='photoz')
        sys.exit(0)
