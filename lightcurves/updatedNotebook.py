filters = ['f140w', 'f105w', 'f814w','f160w']
import sys
import lightcurves
import numpy as np
from IPython.html.widgets import interact, FloatSliderWidget, fixed, FloatTextWidget
from astropy.io import ascii

def parseFile(file = None):
    if file:
        params, my_data = readCSV(file)

        t0 = params['t0']

        data_flux = { f:{'time':[],'flux':[],'flux_error':[]} for f in filters }        
        for f in filters:
            indices = [i for i, x in enumerate(my_data['band']) if x == f]
            data_flux[f]['time'] = [ my_data['mjd'][i] - t0 for i in indices ]
            data_flux[f]['flux'] = [ my_data['flux'][i] for i in indices ]
            data_flux[f]['flux_error'] = [ my_data['fluxerr'][i] for i in indices ]    

        my_dates = [ data_flux['f140w']['time'],
                     data_flux['f105w']['time'],
                     data_flux['f814w']['time']]
        
        createPlots(zValue = params['z'],
                    x0Value = params['x0'],
                    x1Value = params['x1'],
                    cValue = params['c'],
                    my_dates = my_dates,
                    data_flux_filter1 = fixed(data_flux['f140w']['flux']),
                    data_flux_filter1_err = fixed(data_flux['f140w']['flux_error']),
                    data_flux_filter2 = fixed(data_flux['f105w']['flux']),
                    data_flux_filter2_err = fixed(data_flux['f105w']['flux_error']),
                    data_flux_filter3 = fixed(data_flux['f814w']['flux']),
                    data_flux_filter3_err = fixed(data_flux['f814w']['flux_error']),
                    )

    else:
        createPlots(zValue = 1.00,
                x1Value = 0,
		x0Value = None,
                cValue = 0,
                my_dates = np.asarray([[0,32],[0,32],[0,32]]),
                data_flux_filter1=fixed([0.0,4.93]),
                data_flux_filter1_err=fixed([0.26,0.23]),
                data_flux_filter2=fixed([0.0,5.08]),
                data_flux_filter2_err=fixed([0.2,0.2]),
                data_flux_filter3=fixed([0,1.15]),
                data_flux_filter3_err=fixed([0.3,0.3]))

def createPlots(zValue, x0Value, x1Value, cValue, my_dates,
                data_flux_filter1, data_flux_filter1_err,
		data_flux_filter2, data_flux_filter2_err,     
         	data_flux_filter3, data_flux_filter3_err):

    if not x0Value:
        interact(lightcurves.plot_Ia, 
                 data_flux_filter1=data_flux_filter1, data_flux_filter1_err=data_flux_filter1_err,
                 data_flux_filter2=data_flux_filter2, data_flux_filter2_err=data_flux_filter2_err,
                 data_flux_filter3=data_flux_filter3, data_flux_filter3_err=data_flux_filter3_err,
		 z = FloatSliderWidget(min=0.15, max=2, step=0.01, value=zValue),
                 x1 = FloatSliderWidget(min=-3, max=2, step=0.1, value=x1Value),
                 c = FloatSliderWidget(min=-0.4, max=0.4, step=0.01, value=cValue),
                 filters = fixed(filters),
                 dates = fixed(my_dates),
                 phase = FloatSliderWidget(min=-50, max=150, step=1, value=0)); 
    else:
        interact(lightcurves.plot_Ia, 
                 data_flux_filter1=data_flux_filter1, data_flux_filter1_err=data_flux_filter1_err,
                 data_flux_filter2=data_flux_filter2, data_flux_filter2_err=data_flux_filter2_err,
                 data_flux_filter3=data_flux_filter3, data_flux_filter3_err=data_flux_filter3_err,
		 z = FloatSliderWidget(min=0.15, max=2, step=0.01, value=zValue),
                 x0 = FloatSliderWidget(min=0.01*1e-5, max=1e-5, step=0.1*1e-5, value=x0Value),
		 x1 = FloatSliderWidget(min=-3, max=2, step=0.1, value=x1Value),
                 c = FloatSliderWidget(min=-0.4, max=0.4, step=0.01, value=cValue),
                 filters = fixed(filters),
                 dates = fixed(my_dates),
                 phase = FloatSliderWidget(min=-50, max=150, step=1, value=0)); 

def readCSV(file):
    my_data = ascii.read(file, comment = r'\s*@')
    keys = []
    values = []
    for element in my_data.meta['comments']:
        key, value = element.split()
        keys.append(key)
        values.append(float(value))
        
    params = dict(zip(keys, values))

    return params, my_data














