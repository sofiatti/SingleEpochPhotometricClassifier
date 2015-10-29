import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='input_yaml', type=str,
                    help='Input yaml file for this SNe at this epoch')
args = parser.parse_args()
stream = open(args.input_yaml, 'r')
var = yaml.load(stream)

old_to_new_dir = {'mc_files_1000_-10_-5/': 'n1000_phase_-10_-5/',
                  'mc_files_1000_-5_0/': 'n1000_phase_-5_0/',
                  'mc_files_1000_0_5/': 'n1000_phase_0_5/',
                  'mc_files_1000_5_10/': 'n1000_phase_5_10/',
                  'mc_files_10000_-10_-5/': 'n10000_phase_-10_-5/',
                  'mc_files_10000_-5_0/': 'n10000_phase_-5_0/',
                  'mc_files_10000_0_5/': 'n10000_phase_0_5/',
                  'mc_files_10000_5_10/': 'n10000_phase_5_10/'}

if var['photoz_type']['file']['enable'] is not None:
    photoz_type = 'file'
elif var['photoz_type']['gauss']['enable'] is not None:
    photoz_type = 'gauss'
else:
    photoz_type = 'no'

d = {'my_dir': var['my_dir'],
     'simulated_data_dir': old_to_new_dir[var['file_dir']],
     'outdir': var['outdir'],
     'file_with_RF_and_SF_arrays': var['arrays_file'],
     'final_pdf': {'RF': var['final_pdf']['RF'], 'SF': var['final_pdf']['SF'],
                   'photoz': var['final_pdf']['photoz']},
     'photoz_type': photoz_type,
     'file': var['photoz_type']['file']['file_name'],
     'gauss': {'mu': var['photoz_type']['gauss']['mu'],
               'sigma': var['photoz_type']['gauss']['sigma']},
     'photoz_plot_name': 'photoz.pdf',
     'random_forest_plot_name': 'random_forest.pdf',
     'survival_fuction_plot_name': 'survival_function.pdf',
     'final_pdf_plot_name': 'default',
     'filter1': var['filter1'],
     'flux_filter1': var['flux_filter1'],
     'flux_filter1_err': var['flux_filter1_err'],
     'filter2': var['filter2'],
     'flux_filter2': var['flux_filter2'],
     'flux_filter2_err': var['flux_filter2_err'],
     'filter3': var['filter3'],
     'flux_filter3': var['flux_filter3'],
     'flux_filter3_err': var['flux_filter3_err'],
     'x0': 'no'}

new_file = 'new_' + file
with open(new_file, 'w') as yaml_file:
    yaml_file.write(yaml.dump(d, default_flow_style=False))
