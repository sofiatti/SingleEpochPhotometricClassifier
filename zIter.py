import glob
import numpy as np
from PDF import make_cvs, obtain_proba
from kde3d import find_percentile

def iterator(my_dir, file_dir, filter1, filter2, filter3, flux_filter1,
                    flux_filter2, flux_filter3, type=None): 
    files = []
    z = []
    quant = []
    my_files = glob.glob(my_dir + file_dir + '*' + filter1 + '*.gz')
    files.append(sorted(my_files))
    for a in files[0]:
	min_index = a.index('/z') + 2
	z.append(a[min_index:min_index + 3])
    my_z = np.asarray(z, dtype=float)
    my_z = np.divide(my_z, 100)

    z = range(55, 205, 5)
    for i in range(len(files[0])):
	make_csv(my_dir, file_dir, filter1, filter2, filter3, z[i])
	if type == 'pdf':
	    quant.append(obtain_proba(file_dir, flux_filter1, flux_filter2,
			   flux_filter3)[0])
	
	else type =='score':
	    score = find_percentile(my_dir, file_dir, filter1, filter2, filter3,
				flux_filter1, flux_filter2, flux_filter3, z[i])
	    quant.append(score)
    return(quant, my_z)
