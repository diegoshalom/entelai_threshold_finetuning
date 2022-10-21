
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import fcnAFIL
import lesion_classifiers as lc
import time
import SimpleITK as sitk

if os.environ['COMPUTERNAME']=='DIEGO-DESKTOP':
    datadir = 'E:/Entelai/DATA/threshold_finetuning/data completa'
    codedir = 'E:/REPOS/entelai_threshold_finetuning/'
else:
    datadir = 'C:/Entelai/DATA/threshold_finetuning/data completa'
    codedir = 'C:/REPOS/entelai_threshold_finetuning/'
    
outdatadir = os.path.join(datadir, 'out')

PAR = fcnAFIL.build_pairs_theshold_finetuning(datadir)

param = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", \
    'fcn_classify_lesions':lc._classify_lesions_6SD,   \
        'condname':'largesmall_01p_new10'}

t = time.time()
for i, par in enumerate(PAR):
    print('Inicio %s'%(par['studiesname']))

    if len(par['studies'][0]['lesiones_fname'])==0:
        print('Empty filename')
        continue
   
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
    # print(par)
    print("     Fin %s %2.2fs %s" % (par['studiesname'], t1, param['condname']))
    