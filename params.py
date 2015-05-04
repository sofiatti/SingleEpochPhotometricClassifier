import yaml
import sys
from plot import combined

stream = open('input.yaml', 'r')
var = yaml.load(stream)

if sum([var['photoz_type'][0]['file'][0]['enable'],
       var['photoz_type'][1]['gauss'][0]['enable'],
       var['photoz_type'][2]['ignore_photoz']]) != 1:
    sys.exit('You must exatly one type for the photo-z. Gauss, file or '
             'ignore_photoz!')

if var['photoz_type'][0]['file'][0]['enable'] == True:
    if var['photoz_type'][0]['file'][1]['file_name'] == False:
        sys.exit('You must specify a photoz file!')

if var['photoz_type'][1]['gauss'][0]['enable'] == True:
    try:
        val_sigma = float(var['photoz_type'][1]['gauss'][2]['sigma'])
        val_mu = float(var['photoz_type'][1]['gauss'][1]['mu'])
    except ValueError:
        print('Please enter a valid mu or sigma (only floats allowed).')

if var['final_pdf'][2]['photoz'] == True:
    if var['photoz_type'][2]['ignore_photoz'] == True:
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

if var['photoz_type'][0]['file'][0]['enable'] == True:
    photo_z_type = 'file'
    photo_z_file = var['photoz_type'][0]['file'][1]['file_name']
    mu = None
    sigma = None
elif var['photoz_type'][1]['gauss'][0]['enable'] == True:
    photo_z_type = 'gauss'
    photo_z_file = None
    mu = var['photoz_type'][1]['gauss'][1]['mu']
    sigma = var['photoz_type'][1]['gauss'][2]['sigma']
else:
    photo_z_type = None
    photo_z_file = None
    mu = None
    sigma = None
# Change this when you actually see a redshift file!!!
photo_z_redshift_file = None

final_pdf_array = []
if var['final_pdf'][0]['RF'] == True:
    final_pdf_array.append(var['final_pdf'][0].keys()[0])
if var['final_pdf'][1]['SF'] == True:
    final_pdf_array.append(var['final_pdf'][1].keys()[0])
if var['final_pdf'][2]['photoz'] == True:
    final_pdf_array.append(var['final_pdf'][2].keys()[0])

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
