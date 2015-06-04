from load_save import load, get_dict
import numpy as np
filesDir = '/Users/carolinesofiatti/projects/classification/data/test/'
oldFile = load(filesDir + 'original/z145_simulated_mc.gz')
newFile = load(filesDir + 'check/z145_simulated_mc.gz')

filters = ['f105w', 'f140w', 'f160w', 'uvf814w']
fluxes = ['type_Ia_flux', 'type_Ibc_flux', 'type_II_flux']
#params = ['type_Ia_params', 'type_Ibc_params', 'type_II_params']

for item in fluxes:
    oldFluxes = get_dict(oldFile[item])
    for filter in filters:
        if (oldFluxes[filter] == newFile[item[:-5]]['fluxes'][filter]).all():
            print 'OK'
        else:
            print '**ERROR: Fluxes are different. Error to rebuild.'
            print np.shape(oldFluxes[filter])
            print np.shape(newFile[item[:-5]]['fluxes'][filter])

'''
for item in params:
    oldParams = get_dict(oldFile[item])
    for param in oldParams.keys():
        if oldParams[param] == 3:
            print 'OK'
        else:
            print '**ERROR: Params are different. Error to rebuild.'

if oldFile['salt_name'] == newFile['type_Ia']['params']['salt_name']:
    print 'OK'
else:
    print '**ERROR: Fluxes are different. Error to rebuild'

if oldFile['salt_version'] == newFile['type_Ia']['params']['salt_version']:
    print 'OK'
else:
    print '**ERROR: Fluxes are different. Error to rebuild'

'''