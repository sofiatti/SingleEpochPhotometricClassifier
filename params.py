#!/Users/carolinesofiatti/anaconda/bin/python

import yaml
import sys
from plot import combined
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='input_yaml', type=str,
                    help='Input yaml file for this SNe at this epoch')
args = parser.parse_args()
stream = open(args.input_yaml, 'r')
var = yaml.load(stream)

if sum([var['photoz_type']['file']['enable'],
       var['photoz_type']['gauss']['enable'],
       var['photoz_type']['ignore_photoz']]) != 1:
    sys.exit('You must exatly one type for the photo-z. Gauss, file or '
             'ignore_photoz!')

if var['photoz_type']['file']['enable'] == True:
    if var['photoz_type']['file']['file_name'] == False:
        sys.exit('You must specify a photoz file!')

if var['photoz_type']['gauss']['enable'] == True:
    try:
        val_sigma = float(var['photoz_type']['gauss']['sigma'])
        val_mu = float(var['photoz_type']['gauss']['mu'])
    except ValueError:
        print('Please enter a valid mu or sigma (only floats allowed).')

if var['final_pdf']['photoz'] == True:
    if var['photoz_type']['ignore_photoz'] == True:
        sys.exit('If you want the final PDF to have photo-z information, '
                 'the photoz_type cannot be none!')

my_dir = var['my_dir']
file_dir = var['file_dir']
outdir = var['outdir']
arrays_file = var['arrays_file']

filter1 = var['filter1']
filter2 = var['filter2']
filter3 = var['filter3']
flux_filter1 = var['flux_filter1']
flux_filter2 = var['flux_filter2']
flux_filter3 = var['flux_filter3']

if var['photoz_type']['file']['enable'] == True:
    photo_z_type = 'file'
    photo_z_file = var['photoz_type']['file']['file_name']
    mu = None
    sigma = None
elif var['photoz_type']['gauss']['enable'] == True:
    photo_z_type = 'gauss'
    photo_z_file = None
    mu = var['photoz_type']['gauss']['mu']
    sigma = var['photoz_type']['gauss']['sigma']
else:
    photo_z_type = None
    photo_z_file = None
    mu = None
    sigma = None
# Change this when you actually see a redshift file!!!
photo_z_redshift_file = None

final_pdf_array = []
if var['final_pdf']['RF'] == True:
    final_pdf_array.append('RF')
if var['final_pdf']['SF'] == True:
    final_pdf_array.append('SF')
if var['final_pdf']['photoz'] == True:
    final_pdf_array.append('photoz')

final_pdf = ''
i = 0
while i < len(final_pdf_array) - 1:
    final_pdf += final_pdf_array[i]
    final_pdf += '+'
    i += 1
final_pdf += final_pdf_array[i]


combined(final_pdf, my_dir, file_dir, filter1, filter2, filter3,
         flux_filter1, flux_filter2, flux_filter3, outdir, arrays_file,
         photo_z_type, photo_z_file, photo_z_redshift_file, mu, sigma)
