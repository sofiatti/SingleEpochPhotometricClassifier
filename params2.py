import sys
import yaml
import argparse

class yamlData():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(dest='input_yaml', type=str,
                                 help='Input yaml file for this SNe at this epoch')
        self.args = self.parser.parse_args()
        self.stream = open(self.args.input_yaml, 'r')
        self.var = yaml.load(self.stream)
	
	# Directories
	self.my_dir = self.var['my_dir']
	self.simulated_data_dir = self.var['file_dir']
	self.outdir = self.var['outdir']
	
	# Data
	self.filter1 = self.var['filter1']
	self.filter2 = self.var['filter2']
	self.filter3 = self.var['filter3']
	self.flux_filter1 = self.var['flux_filter1']
	self.flux_filter2 = self.var['flux_filter2']
	self.flux_filter3 = self.var['flux_filter3']
	self.flux_filter1_err = self.var['flux_filter1_err']
	self.flux_filter2_err = self.var['flux_filter2_err']
	self.flux_filter3_err = self.var['flux_filter3_err']	
	
	# Salt Parameters
	self.x0 = self.var['x0']
'''
    def plots(self):
	
    def data(self):

    def salt_parameters(self):
''' 
