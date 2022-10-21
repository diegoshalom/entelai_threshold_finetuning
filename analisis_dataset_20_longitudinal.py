# %% Preparo todo, cargo packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import fcnAFIL
import lesion_classifiers as lc
import time
import SimpleITK as sitk
from scipy.optimize import curve_fit
from scipy.stats import norm

if os.environ['COMPUTERNAME']=='DIEGO-DESKTOP':
    datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
    codedir = 'E:/REPOS/entelai_threshold_finetuning/'
else:
    datadir = 'C:/Entelai/DATA/65_MS_Subjects_Longitudinal'
    codedir = 'C:/REPOS/entelai_threshold_finetuning/'
    
outdatadir = os.path.join(datadir, 'out')

# %% Armo y proceso pares y guardo file
param = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", \
    'fcn_classify_lesions':lc.classify_lesion_largesmall_01p_new10, \
        'condname':'largesmall_01p_new10'}

PAR = fcnAFIL.build_PAR_20(datadir)  

t = time.time()
for i, par in enumerate(PAR):
    par = fcnAFIL.full_process_pair(par=par, outdatadir=outdatadir, param=param)

    # Delete some memory-consuming keys, to fit in memory
    if 'index' in par:
        del par['index']    
    for ind, study in enumerate(par['studies']):
        for keyremove in ['img', 'array', 'labels', 'volumen',
                            'lblchanges', 'relabeled']: # for back compatibility
            if keyremove in par['studies'][ind]:
                del par['studies'][ind][keyremove]

    PAR[i] = par

    t1 = time.time() - t
    print("%d/%d %s %2.2fs %s" % (i,len(PAR),par['studiesname'], t1, param['condname']))
print('Listo')
np.savez_compressed(os.path.join(datadir,'PARES_20_long.npz'), PAR=PAR)
# loaded = np.load(os.path.join(datadir,'PARES_20_long.npz'), allow_pickle=True)
# loaded.files
# PAR = list(loaded['PAR'])