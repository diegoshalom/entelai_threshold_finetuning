# %% Load packages 
# region
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from ismember import ismember
from scipy.ndimage import label
import fcnAFIL
from fcnAFIL import list_intersection

import nibabel as nib
import nibabel.processing as processing
import subprocess
import time
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import norm
from scipy.optimize import curve_fit

# from AFIL_to_insert_in_pipeline import AFIL
# from AFIL_to_insert_in_pipeline import calculate_volume_labels
# from AFIL_to_insert_in_pipeline import classify_lesions
import lesion_classifiers as lc

os.chdir('E:\\DIEGO_COMPARTIDO\\OneDrive - dc.uba.ar\\LABO\\Entelai\\AFIL')

# %% Proceso 20 
# region
# Same Reso - Slope 5%: 0.050 (vol$^{1}$)
# Same Reso - Slope 5%: 15.000 (cte sumada)

# Same Reso - Slope 1%: 0.448 (vol$^{1}$)
# Same Reso - Slope 1%: 48.000 (cte sumada)

# param = {"thres":10, "slope":0.05, "growtype":"vol",
#          'fcn_classify_lesions':lc.classify_lesions_old, 'condname':'old'}


param1 = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_1p",  'fcn_classify_lesions':lc.classify_lesion_largesmall_1p,        'condname':'largesmall_1p'}
param2 = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_1p",  'fcn_classify_lesions':lc.classify_lesion_largesmall_1p_new10,  'condname':'largesmall_1p_new10'}
param3 = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", 'fcn_classify_lesions':lc.classify_lesion_largesmall_01p,       'condname':'largesmall_01p'}
param4 = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", 'fcn_classify_lesions':lc.classify_lesion_largesmall_01p_new10, 'condname':'largesmall_01p_new10'}
params = [param1, param2, param3, param4]


# param1 = {"thres":30, "slope":0.0290, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew, 'condname':'1p_nonew_30'}
# param2 = {"thres":40, "slope":0.0115, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew, 'condname':'1p_nonew_40'}
# param3 = {"thres":48, "slope":0.0000, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew, 'condname':'1p_nonew_48'}
# param4 = {"thres":30, "slope":0.0290, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew_new10, 'condname':'1p_nonew_30_10'}
# param5 = {"thres":40, "slope":0.0115, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew_new10, 'condname':'1p_nonew_40_10'}
# param6 = {"thres":48, "slope":0.0000, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesion_nonew_new10, 'condname':'1p_nonew_48_10'}
# params = [param1, param2, param3, param4, param5, param6]

# param1 = {"thres":10, "slope":0.05, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesions_5p_new_vol, 'condname':'5p_new_vol'}
# param2 = {"thres":10, "slope":0.05, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesions_5p_nonew_vol, 'condname':'5p_nonew_vol'}
# param3 = {"thres":10, "slope":0.448, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesions_1p_new_vol, 'condname':'1p_new_vol'}
# param4 = {"thres":10, "slope":0.448, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesions_1p_nonew_vol, 'condname':'1p_nonew_vol'}
# param5 = {"thres":15, "slope":0.00, "growtype":"const", 'fcn_classify_lesions':lc.classify_lesions_5p_new_sum, 'condname':'5p_new_sum'}
# param6 = {"thres":15, "slope":0.00, "growtype":"const", 'fcn_classify_lesions':lc.classify_lesions_5p_nonew_sum, 'condname':'5p_nonew_sum'}
# param7 = {"thres":48, "slope":0.00, "growtype":"const", 'fcn_classify_lesions':lc.classify_lesions_1p_new_sum, 'condname':'1p_new_sum'}
# param8 = {"thres":48, "slope":0.00, "growtype":"const", 'fcn_classify_lesions':lc.classify_lesions_1p_nonew_sum, 'condname':'1p_nonew_sum'}

# params = [param1, param2, param3, param4, param5, param6, param7, param8]

for param in params:
    datadir = 'E:/Entelai/'
    outdatadir = os.path.join(datadir, 'DATA/out/')
    ST = fcnAFIL.build_structure_ST(datadir)
    PAR = fcnAFIL.build_pairs_of_studies(ST)    
    PAR = [par for par in PAR if par['samereso']==True and par['samesubj']==True]        
    
    # datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
    # outdatadir = os.path.join(datadir, 'out/')
    # PAR = fcnAFIL.build_PAR_20(datadir)
    
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
        print("%s %2.2fs %s" % (par['studiesname'], t1, param['condname']))
print('Listo')
# np.savez_compressed(os.path.join(datadir,'PARES_20.npz'), PAR=PAR)
# loaded = np.load(os.path.join(datadir,'PARES_20.npz'), allow_pickle=True)
# loaded.files
# PAR = list(loaded['PAR'])

# %% Armo volumes_all.xlsx, juntando las condiciones

datadir= 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal/21-09-17 dataset 20/'
# datadir= 'E:/Entelai/DATA/21-09-17 dataset control same reso/'

d = os.listdir(datadir) 
dirs = [ name for name in d if os.path.isdir(os.path.join(datadir, name)) ]
for d in dirs:
    for i,param in enumerate(params):
        fname = os.path.join(datadir,d,param['condname'],'volumes.xlsx')
        temp = pd.read_excel(fname)
        temp.rename(columns={'cat': param['condname']}, inplace=True)
        if i==0:
            data = temp.copy()                
        if i>0:
            temp.drop('id', inplace=True, axis=1)
            temp.drop('v1', inplace=True, axis=1)
            temp.drop('v2', inplace=True, axis=1)
            temp.drop('dif', inplace=True, axis=1)
            data = pd.concat([data, temp.reindex(data.index)], axis=1)
    xlsfname = os.path.join(datadir, d, 'volumes_all.xlsx')
    print(xlsfname)
    data.to_excel(xlsfname, index=False)

# %% analizo roll, para estudiar corregistro: Cálculo
datadir = 'E:/Entelai/'
outdatadir = os.path.join(datadir, 'DATA/out/')
ST = fcnAFIL.build_structure_ST(datadir)
PAR = fcnAFIL.build_pairs_of_studies(ST)    
PAR = [par for par in PAR if par['samereso']==True and par['samesubj']==True]        

par=PAR[2]

param = {"thres":30, "slope":0.0290, "growtype":"vol", 'fcn_classify_lesions':lc.classify_lesions_old, 'condname':'100'}
roll = [0, 0, 0]
LTVM, index = fcnAFIL.analyze_roll_shift(par, roll, param)
sum(index['new']) +  sum(index['res']) 


nx, ny, nz = (2, 2, 2)
x = np.linspace(-nx, nx, nx*2+1)
y = np.linspace(-ny, ny, 2*ny+1)
z = np.linspace(-nz, nz, 2*nz+1)
xv, yv, zv = np.meshgrid(x, y, z)
total = np.zeros_like(xv)
t1 = time.time()
for i,xs in enumerate(x):    
    for j,ys in enumerate(y):        
        for k,zs in enumerate(z):
            roll = [int(xv[i,j,k]), int(yv[i,j,k]), int(zv[i,j,k])]
            LTVM, index = fcnAFIL.analyze_roll_shift(par, roll, param)
            total[i,j,k] = sum(index['new']) +  sum(index['res']) 
            # print(xv[i,j,k], yv[i,j,k], zv[i,j,k], total[i,j,k], time.time()-t1)
            print(xs, ys, zs, total[i,j,k], time.time()-t1)

# %% analizo roll, para estudiar corregistro: Gráfico

from matplotlib.patches import Rectangle

plt.figure(1)
for k,valz in zip(list(range(1,4)),z[1:4]):    
    plt.subplot(1,3,k)
    # plt.pcolormesh(x-.5,y-.5,total[:,:,k],vmax=50)
    plt.imshow(total[:,:,k], 
               extent=[x.min()-.5, x.max()+.5, y.min()-.5, y.max()+.5],
               vmin=0, vmax=50)
    # plt.axis([x.min()-.5, x.max()+.5, y.min()-.5, y.max()+.5])    
    plt.xlabel('Roll X [px]')
    plt.ylabel('Roll Y [px]')
    plt.title("# of new+res - Roll Z = %d px"%(valz))
    # plt.title('for different de-corregistrations')
    # plt.colorbar()

    for i,xs in enumerate(x):    
        for j,ys in enumerate(y):        
            if total[i,j,k]>30:
                color = 'k'
            else:
                color = 'w'
            plt.text(xs,ys,int(total[i,j,k]),horizontalalignment='center',verticalalignment='center',color=color)
            
    if valz==0:
        ax = plt.gca()
        rect = Rectangle((-.5,-.5),1,1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

fig = plt.gcf()
fig.set_size_inches(14, 7)


# %% Proceso 6x4
datadir = 'E:/Entelai/'
outdatadir = os.path.join(datadir, 'DATA/out/')
grupos = ["≠suj", "=suj, =reso, =visit", "=suj,  mreso, ≠visit",
          "=suj, ≠reso, =visit", "=suj, ≠reso, ≠visit"]
# endregion
# %% Build ST and PAR
ST = fcnAFIL.build_structure_ST(datadir)
PAR = fcnAFIL.build_pairs_of_studies(ST)

# %% My attempt of AFIL in Python
t = time.time()
cont = 0
for i, par in enumerate(PAR):
    if par['grupo'] == 0:
        t1 = time.time() - t
        print("Skipped %d %2.2fs %2.2fs" % (i, t1, t1 / (cont + 1)))
        continue
    if 'numlabels' in par:
        t1 = time.time() - t
        print("Skipped %d %2.2fs %2.2fs" % (i, t1, t1 / (cont + 1)))
        continue

    # Full process pair    
    par = fcnAFIL.full_process_pair(par=par, outdatadir=outdatadir)
    
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
    print("%d %2.2fs %2.2fs" % (cont, t1, t1 / (cont + 1)))
    cont += 1

#  Para comparar con AFIL Matlab
np.vstack(sorted(PAR[0]['LVTM'], key=lambda x: (x[1], x[2])))

# np.savez_compressed('mifile umbral 30.npz', PAR=PAR)
# np.savez_compressed('mifile umbral 20.npz', PAR=PAR)
# np.savez_compressed('mifile umbral 10.npz', PAR=PAR)
# np.savez_compressed('mifile umbral 0.npz', PAR=PAR)
# np.savez_compressed('mifile umbral 0 diag.npz', PAR=PAR)

# %% Load saved file
# loaded = np.load('mifile umbral 0 diag.npz', allow_pickle=True)
loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

# %% Figuras vol1-vol2 todo junto, e histogramas por categorias
for i,param in enumerate(params):
    # param = param5
    fcn_classify_lesions = param['fcn_classify_lesions']
    
    vol1 = []
    vol2 = []
    for par in PAR:
        if 'LVTM' not in par:
            continue
        item = par['LVTM']
        if item is not np.nan:
            indexcor = np.logical_and(item[:, 1] > -10, item[:, 2] > -10)
            vol1 = np.append(vol1, item[:, 1][indexcor])
            vol2 = np.append(vol2, item[:, 2][indexcor])
    print(len(vol1))
    
    plt.figure(2)
    plt.subplot(2,4,i+1)
    fcnAFIL.figure_vol_vs_vol(vol1, vol2, param)
    plt.xlim([-1, 200])
    plt.ylim([-1, 200])
    plt.title(param['condname'])
    plt.legend().set_visible(False)
    plt.xlabel("")
    plt.ylabel("")

plt.figure(3)
index = fcn_classify_lesions(vol1, vol2)
X = list(range(0,300,5))
categs = ['res', 'new', 'small', 'grow', 'stable']
for i,categ in enumerate(categs):
    plt.subplot(5,1,i+1)
    plt.hist(vol1[index[categ]],X)
    plt.ylabel(categ)

# %%
def v1v2totable(vol1, vol2):
    x = np.array(range(1,1+len(vol1)))
    LVTM = np.transpose(np.stack([x, vol1, vol2])) # Label Volume Tracking Matrix
    LVTM = list(LVTM) # una lista de nparrays
    
    index = lc.classify_lesions_old(vol1, vol2)
    categ = index['new']*1+index['res']*2+index['small']*3+index['grow']*4+index['stable']*5
    categs = ('error','new', 'res', 'small', 'grow', 'stable')    
    
    for i,lv in enumerate(LVTM):
        temp =list(lv)
        temp.append(categs[categ[i]])
        LVTM[i] = temp
        
    df = pd.DataFrame(LVTM, columns=['id', 'v1', 'v2', 'cat'])
    return df

v1v2totable(vol1, vol2)

# %% Comparo conectividad cuadrada con conectividad diagonal
# cargar diag y cargar no diag, y para cada uno hacer hist de tamaño, para new+res, y corr, y small

V=[0,0]
for i, fname in enumerate(['mifile umbral 0.npz', 'mifile umbral 0 diag.npz']):
    loaded = np.load(fname, allow_pickle=True)
    loaded.files
    PAR = list(loaded['PAR'])
    
    v12=[]
    for par in PAR:
        if par['grupo'] == 0:
            continue
        v12 += [[elem[1], elem[2]] for elem in par['LVTM']]
    V[i] = v12


myxor = [x for x in V[0] if x not in V[1]] 
vol1 = [elem[0] for elem in myxor]
vol2 = [elem[1] for elem in myxor]
plt.plot(vol1,vol2,'.', markersize=1)

myxor = [x for x in V[1] if x not in V[0]]
vol1 = [elem[0] for elem in myxor]
vol2 = [elem[1] for elem in myxor]
plt.plot(vol1,vol2,'.', markersize=1)

plt.xscale('log')
plt.yscale('log')
    
print([len(V[0])    , len(V[1])])
    
# %% Lesion size hist all, and Vol1 vs Vol2 in corresponding by group
a = pd.DataFrame(PAR)

plt.figure(1)
newres = a['nnew'].append([a['nresolving']])
cor = a['ncorresponding'].append(a['ncorresponding'])
gruponewres = a['grupo'].append([a['grupo']])
jitter = (np.random.rand(len(gruponewres))-.5)*.1
plt.plot(gruponewres + jitter, newres/(newres + cor) * 100, '.')
plt.ylabel('Percent new+res')
plt.xlabel('Group')
plt.xticks(range(1, 5))
plt.grid(True)

plt.figure(2)
for indg, grupo in enumerate(grupos):
    vol1 = []
    vol2 = []
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if par['grupo'] == indg:
            vol1 += [item[1] for item in par['LVTM'] if item[1] > 0 and
                     item[2] > 0]
            vol2 += [item[2] for item in par['LVTM'] if item[1] > 0 and
                     item[2] > 0]
    if len(vol1) == 0:
        continue
    plt.subplot(2, 3, indg + 1)
    plt.plot(vol1, vol2, '.')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(grupo)
    corr, _ = pearsonr(vol1, vol2)
    print("Correlacion grupo %d: %2.4f" % (indg, corr))


volnew = []
volres = []
volcor = []
for par in PAR:
    if par['grupo'] == 0:
        continue
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    if item is not np.nan:
        index = np.logical_and(item[:, 1] == 0, item[:, 2] > 0)
        vol = item[:, 2][index]
        volnew = np.append(volnew, vol)

        index = np.logical_and(item[:, 1] > 0, item[:, 2] == 0)
        vol = item[:, 1][index]
        volres = np.append(volres, vol)

        index = np.logical_and(item[:, 1] > 0, item[:, 2] > 0)
        vol = item[:, 2][index]
        volcor = np.append(volcor, vol)
plt.figure(3)
plt.subplot(211)
hresnew = plt.hist(np.append(volres, volnew), range(150))
plt.ylabel('Resolving + new')
plt.xlim((0, 60))
plt.ylim((0, 200))
plt.grid(True)
plt.subplot(212)
hcor = plt.hist(volcor, range(150))
plt.ylabel('Corresponding')
plt.xlim((0, 60))
plt.ylim((0, 79))
plt.grid(True)
plt.xlabel('Volume [mm$^3$]')

plt.figure(4)
cumresnew = sum(hresnew[0])-np.cumsum(hresnew[0])
cumcor = sum(hcor[0])-np.cumsum(hcor[0])
plt.plot(cumresnew, '.-', label="resolving + new")
plt.plot(cumcor, '.-', label="corresponding")
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim((0, 60))
# plt.ylim((1, 1000))
plt.grid(True)
plt.xlabel('Threshold [mm$^3$]')
plt.legend()

plt.figure(5)
prcresnew = cumresnew/(cumresnew+cumcor) * 100
plt.plot(prcresnew, '.-', label="resolving + new")
plt.ylabel('Percent')
# plt.yscale('log')
plt.xlim((0, 60))
plt.ylim((0, 14))
plt.grid(True)
plt.xlabel('Threshold [mm$^3$]')
plt.legend()

# %% Lesion size histogram, por grupos
for grupo in range(1, 5):
    volnew = []
    volres = []
    volcor = []
    count = 0
    for par in PAR:
        if par['grupo'] is not grupo:
            continue
        if 'LVTM' not in par:
            continue
        item = par['LVTM']
        if item is not np.nan:
            count += 1
            index = np.logical_and(item[:, 1] == 0, item[:, 2] > 0)
            vol = item[:, 2][index]
            volnew = np.append(volnew, vol)

            index = np.logical_and(item[:, 1] > 0, item[:, 2] == 0)
            vol = item[:, 1][index]
            volres = np.append(volres, vol)

            index = np.logical_and(item[:, 1] > 0, item[:, 2] > 0)
            vol = item[:, 2][index]
            volcor = np.append(volcor, vol)
    plt.figure(3)
    plt.subplot(211)
    hresnew = np.histogram(np.append(volres, volnew), range(150))
    plt.plot(hresnew[0]/count, '.-', label="G%d" % grupo)
    plt.xlim((1, 60))
    plt.ylim([0, max(plt.ylim())])
    plt.text(50, 4, 'Resolving + new', horizontalalignment='center')
    plt.ylabel('Counts / #pairs')
    plt.legend()
    plt.subplot(212)
    hcor = np.histogram(volcor, range(150))
    plt.plot(hcor[0]/count, '.-', label="G%d" % grupo)
    plt.xlim((1, 60))
    plt.ylim([0, max(plt.ylim())])
    plt.text(50, 2.3, 'Corresponding', horizontalalignment='center')
    plt.ylabel('Counts / #pairs')
    plt.xlabel('Volume [mm$^3$]')

    plt.figure(4)
    cumresnew = sum(hresnew[0])-np.cumsum(hresnew[0])
    cumcor = sum(hcor[0])-np.cumsum(hcor[0])
    plt.plot(cumresnew/count, '.-', label="resolving + new G" + str(grupo))
    plt.plot(cumcor/count, '.-', label="corresponding G" + str(grupo))
    plt.ylabel('Counts / #pairs')
    plt.yscale('log')
    plt.xlim((0, 60))
    # plt.ylim((1, 1000))
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.legend()

    plt.figure(5)
    prcresnew = cumresnew/(cumresnew+cumcor) * 100
    plt.plot(prcresnew, '.-', label="resolving + new G" + str(grupo))
    plt.ylabel('Percent')
    # plt.yscale('log')
    plt.xlim((0, 60))
    plt.ylim((0, 20))
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.legend()

# %% Lesion size histogram, Todo, por suj, por reso
subjects = ['VVS', 'FRO', 'PAM', 'CIM', 'TSM', 'GAM']

DATA = []  # Load data
for index, st in ST.iterrows():
    data = {}
    # % Load Lesion masks
    filename = st['lesiones_fname']
    mask = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(mask) != 0
    # data['img'] = img

    # % Label lesion masks
    data['labels'], data['numlabels'] = label(img)
    valores = np.unique(data['labels'])
    y, x = np.histogram(data['labels'].flatten(), valores)
    data['hist'] = y
    data['subject'] = st['subject']
    data['visita'] = st['visita']
    data['AB'] = st['AB']
    data['reso'] = st['reso']
    DATA.append(data)

bars = []
for subject in subjects:
    hist = [data['numlabels'] for data in DATA if data['subject'] == subject]
    bars.append(sum(hist)/4)

plt.figure(1)
x = list(range(6))
plt.bar(x, bars)
plt.xticks(x, subjects)
plt.ylabel('#lesions')


# histogran of sizes of all lesions
plt.figure(1)
hist = [item for data in DATA for item in data['hist'][1:]]
h, x, handle = plt.hist(hist, range(1, 500))
plt.cla()
plt.plot(x[:-1], h, '.-', label='todos')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.xlabel('$Lesion size [mm^3]$')
plt.ylabel('Counts')


# histogran of sizes of lesions by subject
plt.figure(2)
for subject in subjects:
    indexsuj = [index for index, data in enumerate(DATA) if
                data['subject'] == subject]
    datasubj = [DATA[i] for i in indexsuj]
    # datasubj = list(filter(lambda x: x['subject'] == subject,DATA))

    hist = [item for data in datasubj for item in data['hist'][1:]]
    h, x = np.histogram(hist, range(1, 500))
    plt.plot(x[:-1], (h), '.-', label=subject)
    plt.yscale('log')
    plt.xscale('log')
plt.legend()
plt.xlabel('$Lesion size [mm^3]$')
plt.ylabel('Counts')
plt.grid(True)

# histogran of sizes of lesions by resonator
plt.figure(3)
for reso in [2, 4]:
    indexsuj = [index for index, data in enumerate(DATA) if
                data['reso'] == reso]
    datasubj = [DATA[i] for i in indexsuj]
    # datasubj = list(filter(lambda x: x['subject'] == subject,DATA))

    hist = [item for data in datasubj for item in data['hist'][1:]]
    h, x = np.histogram(hist, range(1, 500))
    plt.plot(x[:-1], (h), '.-', label="Resonator " + str(reso))
    plt.yscale('log')
    plt.xscale('log')
plt.legend()
plt.xlabel('$Lesion size [mm^3]$')
plt.ylabel('Counts')
plt.grid(True)

# %% STD vs Lesion size (Todo y por grupo)

vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    if item is not np.nan:
        indexcorr = np.logical_and(item[:, 1] > 0, item[:, 2] > 0)
        vol1 = np.append(vol1, item[:, 1][indexcorr])
        vol2 = np.append(vol2, item[:, 2][indexcorr])
len(vol1)
# temp = vol1
# vol1 = np.append(vol1, vol2)
# vol2 = np.append(vol2, temp)

l1 = sum(np.logical_and(vol1<10, vol2<10 ))
l2 = sum(np.logical_and(vol1>=10, vol2<10 ))
l3 = sum(np.logical_and(vol1<10, vol2>=10 ))
l4 = sum(np.logical_and(vol1>=10, vol2>=10 ))
tot = l1+l2+l3+l4
print([l1,l2,l3,l4])
print("%2.2f %2.2f %2.2f %2.2f" % (l1/tot,l2/tot,l3/tot,l4/tot))

plt.figure(1)
jitterx = np.random.rand(len(vol1))-.5
jittery = np.random.rand(len(vol1))-.5
plt.plot(vol1+0.9*jitterx, vol2+0.9*jittery, '.', markersize=1)
plt.grid(True)
plt.axis('square')
plt.xlabel('Lesion size pre [mm$^3$]')
plt.ylabel('Lesion size post [mm$^3$]')
plt.xlim([0, 50])
plt.ylim([0, 50])

plt.figure(2)
plt.plot(vol1, vol2, '.', markersize=2)
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.axis('square')
plt.xlabel('Lesion size pre [mm$^3$]')
plt.ylabel('Lesion size post [mm$^3$]')

plt.figure(4)
stds = []
values = np.logspace(1, 3, 11)
values = np.linspace(10, 1000, 100)
for value in values:
    index = np.logical_and(vol1 >= value/1.7, vol1 <= value*1.7)
    y = vol2[index]-value
    stds.append(np.std(y))
plt.plot(values, stds, 'o-', label="Data")

plt.plot(values, values*.3, label='vol$^1$')
plt.plot(values, (values**(2/3))*3, label='vol$^{2/3}$')
plt.plot(values, values*0+100, label='vol$^0$')
plt.legend()
plt.xlabel('Lesion size [mm$^3$]')
plt.ylabel('Std Lesion size [mm$^3$]')

for grupo in list(range(1, 5)):

    vol1 = []
    vol2 = []
    for par in PAR:
        if par['grupo'] is not grupo:
            continue
        if 'LVTM' not in par:
            continue
        item = par['LVTM']
        if item is not np.nan:
            indexcorr = np.logical_and(item[:, 1] > 0, item[:, 2] > 0)
            vol1 = np.append(vol1, item[:, 1][indexcorr])
            vol2 = np.append(vol2, item[:, 2][indexcorr])
    len(vol1)
    temp = vol1
    vol1 = np.append(vol1, vol2)
    vol2 = np.append(vol2, temp)

    # plt.figure(3)
    values = np.logspace(1, 3, 11)
    values = np.linspace(10, 1000, 100)
    grow0 = []
    grow1 = []
    grow2 = []
    for value in values:
        index = np.logical_and(vol1 >= value/1.4, vol1 <= value*1.4)
        y = vol2[index]-value
        h = np.histogram(y, range(int(min(y)), int(max(y))))
        # plt.plot(h[1][:-1], h[0]-np.log10(value)*30)
        grow0.append(np.mean(y > 100))
        grow1.append(np.mean(y > (value*.4)))
        grow2.append(np.mean(y > ((value**(2/3))*3)))
    # plt.xlim([-200, 200])

    # plt.figure(5)
    # plt.plot(values, grow0, 'o-', label="0")
    # plt.plot(values, grow1, 'o-', label="1")
    # plt.plot(values, grow2, 'o-', label="2/3")
    # plt.legend()
    # plt.ylim([0, max(plt.ylim())])
    # plt.ylabel('Ratio of labeled "Grow"')
    # plt.xlabel('Lesion size [mm$^3$]')

    plt.figure(5)
    stds = []
    values = np.logspace(1, 3, 11)
    values = np.linspace(1, 1000, 100)
    for value in values:
        index = np.logical_and(vol1 >= value/1.7, vol1 <= value*1.7)
        y = vol2[index]-value
        stds.append(np.std(y))
    plt.plot(values, stds, 'o-', label=grupos[grupo])

plt.plot(values, values*.3, label='vol$^1$')
plt.plot(values, (values**(2/3))*3, label='vol$^{2/3}$')
plt.plot(values, values*0+100, label='vol$^0$')
plt.legend()
plt.xlabel('Lesion size [mm$^3$]')
plt.ylabel('Std Lesion size [mm$^3$]')

# %% Buscando 2SD de la media en lesiones pequeñas
vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    if item is not np.nan:
        indexcor = np.logical_and(item[:, 1] > -10, item[:, 2] > -10)
        vol1 = np.append(vol1, item[:, 1][indexcor])
        vol2 = np.append(vol2, item[:, 2][indexcor])
print(len(vol1))
sizes = range(1, 11)
sd = np.zeros(10)
sd[:] = np.nan
for ind, size in enumerate(sizes):
    index = vol1 == size
    data = vol2[index] - size
    sd[ind - 1] = np.std(data)
plt.plot(sizes, sd, 'o')
plt.xlabel('Size [mm$^3$]')
plt.ylabel('STD [mm$^3$]')


index = vol1 < 11
data = vol2[index]-vol1[index]
sdtotal = np.std(data)
plt.figure(1)
plt.plot(plt.xlim(), [sdtotal, sdtotal])
plt.title('STD total: %2.3f mm$^3$' % (sdtotal))

plt.figure(2)
fcnAFIL.figure_vol_vs_vol(vol1, vol2)
plt.xlim([-1, 200])
plt.ylim([-1, 200])

plt.figure(3)
index = lc.classify_lesions_old(vol1, vol2)
X = list(range(300))
categs = ['res', 'new', 'small', 'grow', 'stable']
for i,categ in enumerate(categs):
    plt.subplot(5,1,i+1)
    plt.hist(vol1[index[categ]],X)
    plt.ylabel(categ)



# %% maximum likelihood, maximizar la probabilidad
#  de observar lo observado, suponiendo que la sd aumenta de tal manera
from scipy.optimize import minimize_scalar
# np.log10(norm.pdf([38.57, 38.58]))


def minus_linkekihood(pend):
    return -sum(np.log10(norm.pdf(x, loc=loc, scale=scale*pend)))


index = lc.classify_lesions_old(vol1, vol2)
index = index['cor']
# index = (vol1 > 10) & (vol2 > 10)
x = vol1[index]
loc = vol2[index]

# Lineal
print("Fit vol^1")
scale = abs(vol2[index])
res = minimize_scalar(minus_linkekihood)
print(res)

# vol^(2/3)
print("Fit vol^(2/3)")
scale = abs(vol2[index])**(2/3)
res = minimize_scalar(minus_linkekihood)
print(res)

# %% false positive rate vs pendiente (vol^1 y vol^2/3 ])

labels = ['Same Reso', 'Different reso', 'All']
for indreso, samereso in enumerate([[True], [False], [True, False]]):
    strreso = labels[indreso]
    vol1 = []
    vol2 = []
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if par['samereso'] not in samereso:
            continue    
        item = par['LVTM']
        if item is not np.nan:
            vol1 = np.append(vol1, item[:, 1])
            vol2 = np.append(vol2, item[:, 2])
                        
    # Me quedo solo con los corresponding
    index = fcnAFIL.classify_lesions(vol1, vol2)
    index = index['cor']
    
    
    # index = np.logical_and(vol1 > 10, vol2 > 10)
    plt.figure(1)
    plt.subplot(3,1,indreso+1)
    pendientes = np.linspace(0, 5, 101)
    fpr = np.zeros_like(pendientes)
    for i, pendiente in enumerate(pendientes):
        fpr[i] = np.mean(vol2[index] > vol1[index] + vol1[index] * pendiente)
    plt.semilogy(pendientes, fpr)
    for i, pendiente in enumerate(pendientes):
        fpr[i] = np.mean(vol1[index] >= vol2[index] + vol2[index] * pendiente)
    plt.semilogy(pendientes, fpr)
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='--')
    plt.ylabel('False positive rate')
    plt.xlabel('Slope (vol$^1$)')
    plt.xlim([0, pendientes[-1]])
    plt.ylim([1e-4, 2])
    # plt.yscale('linear')
    pend1 = pendientes[np.where(fpr < 0.05)[0][0]]
    mystr = "%s - Slope 5%%: %2.3f (vol$^{1}$)" % (strreso ,pend1) 
    print(mystr)
    plt.text(1.5,0.1,mystr)
    # plt.title(mystr)
    plt.plot(plt.xlim(), [0.05, 0.05], 'k--')
    plt.plot([pend1, pend1], plt.ylim(), 'k--')
    
    
    plt.figure(2)
    plt.subplot(3,1,indreso+1)
    pendientes = np.linspace(0, 17, 101)
    fpr = np.zeros_like(pendientes)
    for i, pendiente in enumerate(pendientes):
        fpr[i] = np.mean(vol2[index] >= vol1[index] + (pendiente * (vol1[index] ** (2/3))))
    plt.semilogy(pendientes, fpr)
    for i, pendiente in enumerate(pendientes):
        fpr[i] = np.mean(vol1[index] >= vol2[index] + (pendiente * (vol2[index] ** (2/3))))
    plt.semilogy(pendientes, fpr)
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='--')
    plt.ylabel('False positive rate')
    plt.xlabel('Slope (vol$^{2/3}$)')
    plt.xlim([0, pendientes[-1]])
    plt.ylim([1e-3, 2])
    pend2 = pendientes[np.where(fpr < 0.05)[0][0]]
    mystr = "%s - Slope 5%%: %2.3f (vol$^{2/3}$)" % (strreso, pend2) 
    print(mystr)    
    plt.text(4,0.1,mystr)    
    plt.plot(plt.xlim(),[0.05, 0.05],'k--')
    plt.plot([pend2, pend2], plt.ylim(),'k--')
    
    
    plt.figure(3)
    plt.subplot(1,3,indreso+1)
    jitterx = np.random.rand(sum(index))-.5
    jittery = np.random.rand(sum(index))-.5
    plt.plot(vol1[index]+0.9*jitterx, vol2[index]+0.9*jittery, '.', markersize=1, label='Data')
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    if indreso==0:
        plt.ylabel('Lesion size post [mm$^3$]')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.plot(np.sort(vol1), np.sort(vol1), 'k--')
    plt.plot(np.sort(vol1), np.sort(vol1) + pend1*np.sort(vol1), 'r-', label='~Vol')
    plt.plot(np.sort(vol1), np.sort(vol1) + pend2*np.sort(vol1)**(2/3), 'g-', label='~Sup')
    plt.title(strreso)
    plt.legend()
    
# %% false positive rate vs pendiente (vol^1 y constante ])

alpha = 0.01
labels = ['Same Reso', 'Different reso', 'All']
for indreso, samereso in enumerate([[True], [False], [True, False]]):
    strreso = labels[indreso]
    vol1 = []
    vol2 = []
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if par['samereso'] not in samereso:
            continue    
        item = par['LVTM']
        if item is not np.nan:
            vol1 = np.append(vol1, item[:, 1])
            vol2 = np.append(vol2, item[:, 2])
                        
    # Me quedo solo con los corresponding
    index = lc.classify_lesions_old(vol1, vol2)
    index = np.bitwise_or(index['stable'], index['grow'])
    # Me quedo solo con los no cero
    index = np.bitwise_and(vol1>0,vol2>0)
    vol1=vol1[index]
    vol2=vol2[index]
    
    
    # index = np.logical_and(vol1 > 10, vol2 > 10)
    plt.figure(1)
    plt.subplot(3,1,indreso+1)
    pendientes = np.linspace(0, 5, 2001)
    fpr = np.zeros_like(pendientes)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol2[index] > 10+vol1[index] + vol1[index] * pendiente)
        fpr[i] = np.mean(vol2 > 10+vol1 + vol1 * pendiente)
    plt.semilogy(pendientes, fpr)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol1[index] >= 10+vol2[index] + vol2[index] * pendiente)
        fpr[i] = np.mean(vol1 >= 10+vol2 + vol2 * pendiente)
    plt.semilogy(pendientes, fpr)
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='--')
    plt.ylabel('False positive rate')
    plt.xlabel('Slope (vol$^1$)')
    plt.xlim([0, pendientes[-1]])
    plt.ylim([1e-4, 2])
    # plt.yscale('linear')
    pend1 = pendientes[np.where(fpr < alpha)[0][0]]
    mystr = "%s - Slope %0.0f%%: %2.3f (vol$^{1}$)" % (strreso, alpha*100 ,pend1)     
    print(mystr)
    plt.text(1.5,0.1,mystr)
    # plt.title(mystr)
    plt.plot(plt.xlim(), [0.05, 0.05], 'k--')
    plt.plot([pend1, pend1], plt.ylim(), 'k--')
    
    
    plt.figure(2)
    plt.subplot(3,1,indreso+1)
    pendientes = np.linspace(0, 100, 101)
    fpr = np.zeros_like(pendientes)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol2[index] >= vol1[index] + pendiente )
        fpr[i] = np.mean(vol2 >= vol1 + pendiente )
    plt.semilogy(pendientes, fpr)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol1[index] >= vol2[index] + pendiente )
        fpr[i] = np.mean(vol1 >= vol2 + pendiente )
    plt.semilogy(pendientes, fpr)
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='--')
    plt.ylabel('False positive rate')
    plt.xlabel('Slope (vol$^{2/3}$)')
    plt.xlim([0, pendientes[-1]])
    plt.ylim([1e-3, 2])
    pend2 = pendientes[np.where(fpr < alpha)[0][0]]
    mystr = "%s - Slope %0.0f%%: %2.3f (cte sumada)" % (strreso, alpha*100, pend2) 
    print(mystr)    
    plt.text(4,0.1,mystr)    
    plt.plot(plt.xlim(),[0.05, 0.05],'k--')
    plt.plot([pend2, pend2], plt.ylim(),'k--')
    
    
    plt.figure(3)
    plt.subplot(1,3,indreso+1)    
    jitterx = np.random.rand(len(vol1))-.5
    jittery = np.random.rand(len(vol1))-.5
    plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=1, label='Data')
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    if indreso==0:
        plt.ylabel('Lesion size post [mm$^3$]')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    x=np.sort(vol1)
    plt.plot(x, x, 'k--')
    plt.plot(x, 10+x + pend1*x, 'r-', label='~Vol')
    plt.plot(x, x + pend2, 'g-', label='+Cte')
    plt.title(strreso)
    plt.legend()   
    # plt.xscale('log')
    # plt.yscale('log')
# %% false positive rate vs (pendiente y ordenada) Figuras paper

loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

samereso = [True]
vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    if 'samereso' in par:
        if par['samereso'] not in samereso:
            continue    
    item = par['LVTM']
    if item is not np.nan:
        vol1 = np.append(vol1, item[:, 1])
        vol2 = np.append(vol2, item[:, 2])

# Me quedo solo con los no cero
# index = np.bitwise_and(vol1>0,vol2>0)
# vol1=vol1[index]
# vol2=vol2[index]

vp = (vol2+vol1)/2
dvp = (vol2-vol1)/vp
vp = np.delete(vp, np.bitwise_or(vol1==0,vol2==0))  #borro los no corresponding
dvp = np.delete(dvp,np.bitwise_or(vol1==0,vol2==0))  #borro los no corresponding

# temp = pd.DataFrame(list(zip(vol1,vol2,vp,dvp)),columns=["vol1","vol2","vp","dvp"])
# xlsfname = os.path.join('test.xlsx')
# temp.to_excel(xlsfname, index=False)

plt.figure(10)
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.subplot(2,1,1)
plt.hist(dvp,100)
plt.xlabel('Relative change in volume')
plt.ylabel('Counts')
plt.subplot(2,1,2)
plt.plot(dvp,vp,'.')
plt.xlabel('Relative change in volume')
plt.ylabel('Mean volume [mm$^3$]')
# plt.ylim([0, 200])

print("El desvío estandar de todas es %2.4f"%(np.std(dvp)))
umbral = 25
print("El desvío estandar de las pequeñas (<%d) es %2.4f"%(umbral, np.std(dvp[vp<umbral])))
print("El desvío estandar de las grandes (>%d) es %2.4f"%(umbral, np.std(dvp[vp>=umbral])))
umbral = 50
print("El desvío estandar de las pequeñas (<%d) es %2.4f"%(umbral, np.std(dvp[vp<umbral])))
print("El desvío estandar de las grandes (>%d) es %2.4f"%(umbral, np.std(dvp[vp>=umbral])))


alpha = 0.01
thresholds = [10, 20, 30, 40, 48]
slopes = []
for ind, thres in enumerate(thresholds):
    
    plt.figure(1)
    # plt.subplot(3,1,indreso+1)
    pendientes = np.linspace(0, 1, 2001)
    fpr = np.zeros_like(pendientes)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol2[index] > 10+vol1[index] + vol1[index] * pendiente)
        fpr[i] = np.mean(vol2 > thres+vol1 + vol1 * pendiente)
    plt.semilogy(pendientes, fpr)
    for i, pendiente in enumerate(pendientes):
        # fpr[i] = np.mean(vol1[index] >= 10+vol2[index] + vol2[index] * pendiente)
        fpr[i] = np.mean(vol1 >= thres+vol2 + vol2 * pendiente)
    plt.semilogy(pendientes, fpr)
    plt.grid(True)
    plt.grid(b=True, which='minor', linestyle='--')
    plt.ylabel('False positive rate')
    plt.xlabel('Slope (vol$^1$)')
    plt.xlim([0, pendientes[-1]])
    plt.ylim([1e-4, 2])
    # plt.yscale('linear')
    pend1 = pendientes[np.where(fpr < alpha)[0][0]]
    slopes.append(pend1)
    # mystr = "%s - Slope %0.0f%%: %2.3f (vol$^{1}$)" % (strreso, alpha*100 ,pend1)     
    # print(mystr)
    # plt.text(1.5,0.1,mystr)
    # plt.title(mystr)
    plt.plot(plt.xlim(), [0.05, 0.05], 'k--')
    plt.plot([pend1, pend1], plt.ylim(), 'k--')        
    
    plt.figure(3)
    x=np.sort(np.hstack((vol1,vol2)))
    if ind == 0:
        jitterx = np.random.rand(len(vol1))-.5
        jittery = np.random.rand(len(vol1))-.5
        plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
        plt.plot(x, x, 'k--',linewidth=0.5)
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    plt.ylabel('Lesion size post [mm$^3$]')    
    
    plt.plot(x, thres + x + pend1*x, label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=0.5)
    # plt.legend()   
    plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0.25))
    # plt.xlim([0, 100])
    # plt.ylim([0, 100])
    # plt.xlim([0, max(x)])
    # plt.ylim([0, max(x)])
    
    plt.xscale('log')
    plt.yscale('log')


plt.figure(5)
thresholds, slopes = [[48], [0]]
for ind, (thres, pend1) in enumerate(zip(thresholds, slopes)):    
        
    x=np.sort(np.hstack((vol1,vol2)))
    if ind == 0:
        jitterx = np.random.rand(len(vol1))-.5
        jittery = np.random.rand(len(vol1))-.5
        plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
        plt.plot(x, x, 'k--',linewidth=0.5)
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    plt.ylabel('Lesion size post [mm$^3$]')    
    
    linewidth=1.5
    
    x = np.linspace (0,max(x)*1.1,1000)
    plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
    plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
    # plt.legend()   
    # plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0.25))
    plt.xlim([0, 500])
    plt.ylim([0, 500])
    # plt.xlim([0, max(x)])
    # plt.ylim([0, max(x)])

    # plt.xlim([.8, 3200])
    # plt.ylim([.8, 3200])    
    # plt.xscale('log')
    # plt.yscale('log')


####    Cargo el otro dataset #######################################
datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
loaded = np.load(os.path.join(datadir,'PARES_20.npz'), allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

samereso = [True]
vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    if item is not np.nan:
        vol1 = np.append(vol1, item[:, 1])
        vol2 = np.append(vol2, item[:, 2])

plt.figure(6)
thresholds, slopes = [[48], [0]]
for ind, (thres, pend1) in enumerate(zip(thresholds, slopes)):    
        
    x=np.sort(np.hstack((vol1,vol2)))
    if ind == 0:
        jitterx = np.random.rand(len(vol1))-.5
        jittery = np.random.rand(len(vol1))-.5
        plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
        plt.plot(x, x, 'k--',linewidth=0.5)
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    plt.ylabel('Lesion size post [mm$^3$]')    
    
    linewidth=1.5
    
    x = np.linspace (0,max(x)*1.1,10000)
    plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
    plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
    # plt.legend()   
    # plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0.25))
    plt.xlim([0, 500])
    plt.ylim([0, 500])
    # plt.xlim([0, max(x)])
    # plt.ylim([0, max(x)])

    # plt.xlim([.8, 32000])
    # plt.ylim([.8, 32000])    
    # plt.xscale('log')
    # plt.yscale('log')


loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

samereso = [True]
vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    if 'samereso' in par:
        if par['samereso'] not in samereso:
            continue    
    item = par['LVTM']
    if item is not np.nan:
        vol1 = np.append(vol1, item[:, 1])
        vol2 = np.append(vol2, item[:, 2])

plt.figure(100)
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.subplot(2,2,1)
plt.text(10,440,'a',fontsize=20)
thres, pend, ind = [48, 0, 0]
x=np.sort(np.hstack((vol1,vol2)))
jitterx = np.random.rand(len(vol1))-.5
jittery = np.random.rand(len(vol1))-.5
plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
plt.plot(x, x, 'k--',linewidth=0.5)
plt.grid(True)
plt.axis('square')
# plt.xlabel('Lesion size pre [mm$^3$]')
plt.ylabel('Lesion size post [mm$^3$]')    
linewidth=1.5
x = np.linspace (0,max(x)*1.1,1000)
plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
plt.xlim([0, 500])
plt.ylim([0, 500])

plt.subplot(2,2,2)
plt.text(1,1150,'b',fontsize=20)
x=np.sort(np.hstack((vol1,vol2)))
jitterx = np.random.rand(len(vol1))-.5
jittery = np.random.rand(len(vol1))-.5
plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
plt.plot(x, x, 'k--',linewidth=0.5)
plt.grid(True)
plt.axis('square')
# plt.xlabel('Lesion size pre [mm$^3$]')
# plt.ylabel('Lesion size post [mm$^3$]')    
linewidth=1.5
x = np.linspace (0,max(x)*1.1,1000)
plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
# plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
plt.xlim([.8, 3200])
plt.ylim([.8, 3200])    
plt.xscale('log')
plt.yscale('log')

datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
loaded = np.load(os.path.join(datadir,'PARES_20.npz'), allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])


samereso = [True]
vol1 = []
vol2 = []
for par in PAR:
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    if item is not np.nan:
        vol1 = np.append(vol1, item[:, 1])
        vol2 = np.append(vol2, item[:, 2])

plt.subplot(2,2,3)
plt.text(10,450,'c',fontsize=20)
thres, pend, ind = [48, 0, 0]
x=np.sort(np.hstack((vol1,vol2)))
jitterx = np.random.rand(len(vol1))-.5
jittery = np.random.rand(len(vol1))-.5
plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
plt.plot(x, x, 'k--',linewidth=0.5)
plt.grid(True)
plt.axis('square')
plt.xlabel('Lesion size pre [mm$^3$]')
plt.ylabel('Lesion size post [mm$^3$]')    
linewidth=1.5
x = np.linspace (0,max(x)*1.1,1000)
plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
plt.xlim([0, 500])
plt.ylim([0, 500])

plt.subplot(2,2,4)
plt.text(1,10000,'d',fontsize=20)
x=np.sort(np.hstack((vol1,vol2)))
jitterx = np.random.rand(len(vol1))-.5
jittery = np.random.rand(len(vol1))-.5
plt.plot(vol1+0.*jitterx, vol2+0.*jittery, '.', markersize=3, label='Data')        
plt.plot(x, x, 'k--',linewidth=0.5)
plt.grid(True)
plt.axis('square')
plt.xlabel('Lesion size pre [mm$^3$]')
# plt.ylabel('Lesion size post [mm$^3$]')    
linewidth=1.5
x = np.linspace (0,max(x)*1.1,10000)
plt.plot(x, thres + x + pend1*x, 'r',label="cte=%d, pend=%2.4f"%(thres, pend1),linewidth=linewidth)
# plt.plot([0,0], [10,thres],'r',linewidth=linewidth, clip_on=False,  zorder=10)
plt.xlim([.8, 32000])
plt.ylim([.8, 32000])    
plt.xscale('log')
plt.yscale('log')

# %% Incerteza en el volumen total
loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

samereso = [True]
for par in PAR:
    if 'LVTM' not in par:
        continue
    # if 'samereso' in par:
    #     if par['samereso'] not in samereso:
    #         continue    
    item = par['LVTM']
    par['vol1'] = sum(item[:, 1])
    par['vol2'] = sum(item[:, 2])
    par['dv'] = par['vol2']-par['vol1']
    par['vp'] = (par['vol2']+par['vol1'])/2
    par['dvp'] = par['dv']/par['vp']*100

v1 = [par['vol1'] for par in PAR if 'dvp' in par]
v2 = [par['vol2'] for par in PAR if 'dvp' in par]

vp = [par['vp'] for par in PAR if 'dvp' in par]
vp = np.array(vp)

dvp = [par['dvp'] for par in PAR if 'dvp' in par]
dvp = np.array(dvp)

dv = [par['dv'] for par in PAR if 'dvp' in par]
dv = np.array(dv)

samereso = [par['samereso'] for par in PAR if 'dvp' in par]
x = np.linspace(-40,40,17)
plt.figure(1)
h1 = plt.hist(dvp[samereso],x)
h2 = plt.hist(dvp[np.bitwise_not(samereso)],x)
plt.figure(2)
fig = plt.gcf()
fig.set_size_inches(8, 8)
ax1 = plt.subplot(2,1,2)
ancho = 2
plt.bar(h1[1][:-1]-ancho/2,h1[0],ancho,label='same reso')
# plt.plot(h1[1][:-1]-1,h1[0])
plt.bar(h2[1][:-1]+ancho/2,h2[0],ancho,label='diff reso')
# plt.plot(h2[1][:-1]+1,h2[0])
    
plt.xlabel('% volumen change')
plt.ylabel('Counts')
plt.legend()
plt.grid(True)
plt.xlim([-30,40])

std1 = np.std(dvp[samereso])
m1 = np.mean(dvp[samereso])
std2 = np.std(dvp[np.bitwise_not(samereso)])
m2 = np.mean(dvp[np.bitwise_not(samereso)])

plt.plot([m1-std1,m1+std1],[3.5,3.5])
plt.plot( [m2-std2,m2+std2], [5.5,5.5])
plt.text(10,3.4,'SD = %2.2f%%'%(std1))
plt.text(20,5.4,'SD = %2.2f%%'%(std2))

plt.subplot(2,1,1, sharex=ax1)
index = samereso
plt.scatter(dvp[index],vp[index],label='same reso')
index = np.bitwise_not(samereso)
plt.scatter(dvp[index],vp[index],label='diff reso')

plt.xlabel('% volumen change')
plt.ylabel('Mean volume [mm$^3$]')
plt.legend()
plt.grid(True)


plt.figure(8)
x = np.linspace(-40,40,17)
h1 = plt.hist(dvp[samereso],x)
h2 = plt.hist(dvp[np.bitwise_not(samereso)],x)
plt.clf()
ancho = 2
plt.bar(h1[1][:-1]-ancho/2,h1[0],ancho,label='SameReso')
# plt.plot(h1[1][:-1]-1,h1[0])
plt.bar(h2[1][:-1]+ancho/2,h2[0],ancho,label='DiffReso')
# plt.plot(h2[1][:-1]+1,h2[0])
    
plt.xlabel('% volumen change')
plt.ylabel('Counts')
plt.legend()
plt.grid(True)
plt.xlim([-30,40])

std1 = np.std(dvp[samereso])
m1 = np.mean(dvp[samereso])
std2 = np.std(dvp[np.bitwise_not(samereso)])
m2 = np.mean(dvp[np.bitwise_not(samereso)])

plt.plot([m1-std1,m1+std1],[3.5,3.5])
plt.plot( [m2-std2,m2+std2], [5.5,5.5])
plt.text(10,3.4,'SD = %2.2f%%'%(std1))
plt.text(20,5.4,'SD = %2.2f%%'%(std2))



plt.figure(9)
plt.plot(v1,v2,'.')
plt.axis('square')
myylim =15000
plt.plot( [-10000,myylim],[-10000, myylim],color=[.5,.5,.5],linewidth=0.5)
plt.grid(True)
plt.xlabel('V$_1$ [mm$^3$]')
plt.ylabel('V$_2$ [mm$^3$]')


datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
loaded = np.load(os.path.join(datadir,'PARES_20.npz'), allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

for par in PAR:
    if 'LVTM' not in par:
        continue
    item = par['LVTM']
    par['vol1'] = sum(item[:, 1])
    par['vol2'] = sum(item[:, 2])
    par['dv'] = par['vol2']-par['vol1']
    par['vp'] = (par['vol2']+par['vol1'])/2
    par['dvp'] = par['dv']/par['vp']*100

vp = [par['vp'] for par in PAR if 'dvp' in par]
vp = np.array(vp)

dvp = [par['dvp'] for par in PAR if 'dvp' in par]
dvp = np.array(dvp)
x = np.linspace(-40,40,17)
plt.figure(4)
h1 = plt.hist(dvp,x)
plt.figure(5)
ancho = 2
plt.bar(h1[1][:-1]-ancho/2,h1[0],ancho)
    
plt.xlabel('% volumen change')
plt.ylabel('Counts')
plt.legend()
plt.grid(True)

plt.figure(6)
index = samereso
plt.scatter(dvp,vp)

plt.xlabel('% volumen change')
plt.ylabel('Mean volume [mm$^3$]')
plt.grid(True)

for par in PAR:
    plt.text(par['dvp']+2,par['vp'],par['studiesname'])

fig = plt.gcf()
fig.set_size_inches(8, 8)
# plt.plot([0,1*std1], [26000,26000])
plt.plot([0,2*std1], [24000,24000])
# plt.plot([0,3*std1], [22000,22000])
# plt.text(-7,25800,'1 SD')
plt.text(-7,23800,'2 SD')
# plt.text(-7,21800,'3 SD')
myylim =plt.ylim()
# plt.plot([1*std1,1*std1], myylim ,color=[.5,.5,.5])
plt.plot([2*std1,2*std1], myylim ,color=[.5,.5,.5])
# plt.plot([3*std1,3*std1], myylim ,color=[.5,.5,.5])
plt.ylim(myylim)

# %% Incerteza de lesiones individuales

def extract_vp_dvp(PAR):
    vol1 = np.array([])
    vol2 = np.array([])
    samereso = [True]
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if 'samereso' in par:
            if par['samereso'] not in samereso:
                continue  
        item = par['LVTM']
        vol1 = np.append(vol1,item[:,1])
        vol2 = np.append(vol2,item[:,2])
    
    index = np.bitwise_or(vol1==0,vol2==0)
    vol1 = vol1[~index]
    vol2 = vol2[~index]
    
    vp = (vol2+vol1)/2.
    dvp = (vol2-vol1)/vp*100

    zipped = list(zip(vp,dvp,vol1,vol2))
    res = sorted(zipped, key = lambda x: x[0])
    vp = [item[0] for item in res ]
    dvp = [item[1] for item in res ]
    vol1 = [item[2] for item in res ]
    vol2 = [item[3] for item in res ]

    return vp,dvp,vol1,vol2

def plotv1v2(v1,v2,maxx):
    plt.plot(v1,v2,'.')
    plt.axis('square')
    plt.xlim([0,maxx])
    plt.ylim([0,maxx])
    plt.grid(True)
    x = np.linspace (0,maxx,1000)
    plt.plot(x, x, 'k--',linewidth=0.5)
    plt.xlabel('V$_1$ [mm$^3$]')
    plt.ylabel('V$_2$ [mm$^3$]')

def plotdvpdv(v1,v2):
    vp = (v2+v1)/2.
    dvp = (v2-v1)/vp*100
    plt.plot(dvp,vp,'.')
    plt.xlim([-200,200])
    plt.grid(True)
    plt.xlabel("% change volume")
    plt.ylabel("Mean volume [mm$^3$]")

    
def plot_v1v2_dvpdv(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    maxx = max(max(v1),max(v2))*1.05
    
    plt.figure()
    h=dict()
    h['fig'] = plt.gcf()
    h['fig'].set_size_inches(7, 7)    
    
    h['h1'] = plt.subplot(221)
    plotv1v2(v1,v2,maxx)

    h['h2'] = plt.subplot(222)
    plotv1v2(v1,v2,maxx*1.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([.7,maxx*1.3])
    plt.ylim([.7,maxx*1.3])    

    h['h3'] = plt.subplot(223)
    plotdvpdv(v1,v2)

    h['h4'] = plt.subplot(224)
    plotdvpdv(v1,v2)
    plt.yscale('log')
    
    plt.tight_layout()
    
    return h

def plot_v1v2_4escalas(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    maxx = max(max(v1),max(v2))*1.05
    
    plt.figure()
    h=dict()
    h['fig'] = plt.gcf()
    h['fig'].set_size_inches(7, 7)    
    
    h['h1'] = plt.subplot(221)
    plotv1v2(v1,v2,maxx)
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    h['h2'] = plt.subplot(222)
    plotv1v2(v1,v2,maxx)
    plt.xlim([0, 250])
    plt.ylim([0, 250])

    h['h3'] = plt.subplot(223)
    plotv1v2(v1,v2,maxx)
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    
    h['h4'] = plt.subplot(224)
    plotv1v2(v1,v2,maxx)
    
    plt.tight_layout()
    return h

def AgregoCte(h1s,h2s):
    for ax in h1s:
        plt.sca(h[ax])
        maxx = max(plt.ylim())
        x = np.linspace (0,maxx,1000)
        x = np.logspace(-1,5,100)
        plt.plot(x, x + 48, 'r',label="cte=48")
        
    for ax in h2s:
        plt.sca(h[ax])
        ylim = plt.ylim()
        x = np.logspace(-1,5,100)
        plt.plot(100*1/((x/48)+.5),x,'r',linewidth=2)
        plt.ylim(ylim)    

def Agrego2sdSmallLarge(h1s,h2s,umbral, sd1,sd2):    
    for ax in h1s:
        plt.sca(h[ax])
        maxx = max(plt.ylim())
        
        x = np.linspace (0,umbral,1000)
        y = x*(1+nstd*sd1/100)
        plt.plot(x, y, 'g')
        plt.plot([umbral, umbral],[umbral,umbral*(1+nstd*sd1/100)],'--g')
        plt.plot([umbral, umbral],[0,maxx],'--g')
        x = np.linspace (umbral,maxx,1000)
        y = x*(1+nstd*sd2/100)
        plt.plot(x, y, 'g',label="sd(%Vc) small/large")
    
    for ax in h2s:
        plt.sca(h[ax])
        ylim = plt.ylim()
    
        plt.plot([-nstd*sd1,nstd*sd1],[umbral, umbral],'--g')
        plt.plot([nstd*sd1,nstd*sd1],[.1, umbral],'g')
        plt.plot([-nstd*sd1,-nstd*sd1],[.1, umbral],'g')
        plt.plot([nstd*sd2,nstd*sd2],[1e4, umbral],'g')
        plt.plot([-nstd*sd2,-nstd*sd2],[1e4, umbral],'g')
        plt.ylim(ylim)  
        
def Agrego2sdExpFit(h1s,h2s,popt):    
    for ax in h1s:
        plt.sca(h[ax])
        maxx = max(plt.ylim())
        x = np.linspace (0,maxx,1000)
        plt.plot(x, x*(1+nstd*func(x,*popt)/100), 'c',linewidth=2,label="Exp fit of in sd(%Vc)")
    
    for ax in h2s:
        plt.sca(h[ax])
        maxx = max(plt.ylim())
        
        x=range(0,int(maxx))
        plt.plot(nstd*func(x,*popt),x, 'c',linewidth=2)

def AgregoRecta(h1s,h2s,P,pend):    
    v1 = np.linspace (0,65,66)
    v2 = P + pend*v1
    vp = (v2+v1)/2.
    dvp = (v2-v1)/vp*100    
    for ax in h1s:
        plt.sca(h[ax])        
        plt.plot(v1,v2,'r')        
    
    for ax in h2s:
        plt.sca(h[ax])             
        plt.plot(dvp,vp, 'r',linewidth=2)

###########################################
###########################################

loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

nstd = 2.32634
nstd = 3
# nstd = 4


# datadir = 'E:/Entelai/DATA/20/65_MS_Subjects_Longitudinal'
# loaded = np.load(os.path.join(datadir,'PARES_20.npz'), allow_pickle=True)
# loaded.files
# PAR = list(loaded['PAR'])

vp, dvp, vol1, vol2 = extract_vp_dvp(PAR)

###########################################
### Armo tabla sd(%Vc) small/large

tabla=[]
for umbral in [10,20,30,40,48,50,65,100,200,300,400,500,1000]:
    # umbral = 50
    data1 = [val[1] for val in zip(vp,dvp) if val[0]<=umbral]
    sd1 = np.std(data1)
    data2 = [val[1] for val in zip(vp,dvp) if val[0]>umbral]
    sd2 = np.std(data2)    
    print("%d %2.4f (%d) %2.4f (%d)"%(umbral,sd1,len(data1),sd2,len(data2)))
    tabla.append({'thres':umbral,'sd1':sd1,'n1':len(data1),'sd2':sd2,'n2':len(data2)})

umbral = 65
tablaumbral = [item for item in tabla if item['thres']==umbral ][0]

##########  Ajuste exponencial STD de %VC to mean volume
plt.figure(100)
ventana = 50
nventanas = 100
x=np.zeros(nventanas)
x[:] = np.nan
y=np.zeros(nventanas)
x[:] = np.nan
for i in range(nventanas):
    desde = int(i*np.floor((len(vp)-ventana)/nventanas))
    hasta = desde + ventana        
    x[i] = np.mean(vp[desde:hasta])
    y[i] = np.std(dvp[desde:hasta])
    
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

maxx = max(x)*1.05 
plt.plot(x,y,'.',label='data')    
popt, pcov = curve_fit(func, x, y)
plt.plot(x,y-func(x, *popt),'.')
print("el desvío de los residuos es %g" % np.std(y-func(x, *popt)))
print("valor de la exonencial en 3tau es %g" % (float(func(65, *popt))-float(popt[2])))

p_sigma = np.sqrt(np.diag(pcov))
x = np.linspace(0,maxx,1000)
plt.plot(x, func(x, *popt), 'c-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.xlabel('Mean volume [mm$^3$]')
plt.ylabel('stdev of %Vc')
plt.grid(True)
plt.xlim([0 ,maxx])
plt.title('Umbral = 3/b = %2.3f mm$^3$' %  (3/popt[1]))
umbral = np.floor(3/popt[1])
myylim = plt.ylim()
plt.plot([umbral, umbral],myylim,'--g')
plt.ylim(myylim)


plt.figure(101)
ventana = 50
nventanas = 100
x=np.zeros(nventanas)
x[:] = np.nan
y=np.zeros(nventanas)
x[:] = np.nan
for i in range(nventanas):
    desde = int(i*np.floor((len(vp)-ventana)/nventanas))
    hasta = desde + ventana        
    x[i] = np.mean(vp[desde:hasta])
    y[i] = np.std(dvp[desde:hasta])
    
maxx = max(x)*1.05 
plt.plot(x,y,'.',label='data')    
x = np.linspace(0,maxx,1000)



sd1 = tablaumbral['sd1']
# plt.plot([0,umbral],[sd1, sd1],'g')
sd2 = tablaumbral['sd2']
plt.plot([umbral,maxx],[sd2,sd2],'g')
ylim = plt.ylim()
plt.plot([umbral,umbral],plt.ylim(),'--g')
plt.ylim(ylim) 

plt.xlabel('Mean volume [mm$^3$]')
plt.ylabel('stdev of %Vc')
plt.grid(True)
plt.xlim([0 ,maxx])
plt.title('Factor: %2.3f%%'%(sd2))
#####################################


        
        
h = plot_v1v2_dvpdv(vol1,vol2)
AgregoCte(['h1','h2'],['h3','h4'])
Agrego2sdSmallLarge(['h1','h2'],['h3','h4'],umbral, tablaumbral['sd1'],tablaumbral['sd2'])
Agrego2sdExpFit(['h1','h2'],['h3','h4'],popt) 

h = plot_v1v2_4escalas(vol1,vol2)
AgregoCte(['h1','h2','h3','h4'],[])
Agrego2sdSmallLarge(['h1','h2','h3','h4'],[],umbral, tablaumbral['sd1'],tablaumbral['sd2'])
Agrego2sdExpFit(['h1','h2','h3','h4'],[],popt) 



mysd = np.zeros(int(umbral))
vol1=np.array(vol1)
vol2=np.array(vol2)
for i in range(1,int(umbral)+1):
    data = np.concatenate((vol2[vol1==i], vol1[vol2==i]))
    mysd[i-1] = np.std(data) 

plt.figure(4)
plt.subplot(2,1,1)
x = np.linspace(1,int(umbral),int(umbral))
plt.plot(x,mysd)
plt.ylabel('Stdev [mm$^3$]')
plt.xlabel('Volume [mm$^3$]')

plt.subplot(2,1,2)
h = plt.hist(vol1+vol2,range(0,1+int(umbral)))
plt.cla()
plt.bar(h[1][:-1],h[0]/len(h[0]),width=1)
plt.xlabel('Volume [mm$^3$]')
plt.ylabel('Freq')

peso = h[0]/sum(h[0])

pend_arr = (1+nstd*tablaumbral['sd2']/100)

oo = umbral*pend_arr - umbral
# La oo la obtengo moviendo el valor para el nstd correspondiente, hasta obtener que val1 == val2
if nstd == 4:
    oo = 25.365 
elif nstd == 3:
    oo = 19 
else:
    oo = 14.7
    
U1 = umbral
U2 = umbral*(1+nstd*tablaumbral['sd2']/100)
pend = (U2-oo)/umbral

recta = oo+pend*x
fraccarr = (1-norm.cdf(recta,x,np.mean(mysd)))
value1 = sum(peso * fraccarr)
value2 = 1-norm.cdf(nstd)

h = plot_v1v2_4escalas(vol1,vol2)
Agrego2sdSmallLarge(['h1','h2','h3','h4'],[],umbral, np.nan,tablaumbral['sd2'])
AgregoRecta(['h1','h2','h3','h4'],[],oo,pend)


print('Ecuación de la recta 1: V_2C = %g + V1 * %g'%(oo,pend))
print('Ecuación de la recta 2: V_2C = V1 * %g'%(pend_arr))
print('val1 = %g%% \nval2 = %g%%'%(value1*100,value2*100))





# %% incerteza de lesiones new/resolving
loaded = np.load('mifile umbral 0.npz', allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

def extract_newres(PAR):
    vol1 = np.array([])
    vol2 = np.array([])
    samereso = [True]
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if 'samereso' in par:
            if par['samereso'] not in samereso:
                continue  
        item = par['LVTM']
        vol1 = np.append(vol1,item[:,1])
        vol2 = np.append(vol2,item[:,2])
    
    index = np.bitwise_or(vol1==0,vol2==0)
    vol1 = vol1[index]
    vol2 = vol2[index]
    vol1 = vol1[vol1>0]
    vol2 = vol2[vol2>0]
    
    return vol1,vol2

vol1,vol2 = extract_newres(PAR)

vol = np.concatenate([vol1,-vol2])

h=plt.hist(vol,range(-30,30))
plt.clf()
plt.bar(h[1][:-1],h[0],width=1)
plt.xlabel('Volume of new/resolving [mm$^3$]')
plt.ylabel('Counts')
vol=sorted(vol)
vol =np.array(vol)
vol = vol[np.abs(vol)<40.0]
mystd = np.std(vol)
mymean = np.mean(vol)
print("El desvío estándar es: %2.3f mm3"%mystd )
print("Para 1%% (2.3264 SD): : %2.3f mm3"%(mystd *2.3264))
print("Para 0.1%% (3 SD): : %2.3f mm3"%(mystd *3))
print("Para 4 SD: : %2.3f mm3"%(mystd *4))

print(mymean )
print(norm.cdf(2.3264))
print(1-norm.cdf(3))

x=range(-20,21)
plt.plot(x, len(vol)*norm.pdf(x,mymean,mystd), 'k-', lw=2, label='frozen pdf')
plt.xlim([-20,20])
plt.title("SD = %2.3f mm3"%mystd )
plt.grid(True)


# %% Stats number of lesions samereso, cor, res, new
labels = ['Same Reso', 'Different reso', 'All']
for indreso, samereso in enumerate([[True], [False], [True, False]]):
    strreso = labels[indreso]
    print(samereso)
    for par in PAR:
        if 'LVTM' not in par:
            continue
        if par['samereso'] not in samereso:
            continue    
        item = par['LVTM']
        if item is not np.nan:            
            vol1 = item[:, 1]            
            vol2 = item[:, 2]                        
    
        index = fcnAFIL.classify_lesions(vol1, vol2)
        resnew = sum(index['res']) + sum(index['new'])
        cor = sum(index['cor'])
        print(par['studiesname'], cor, resnew)

# %% multiple converging and separating labels
for indp, par in enumerate(PAR):
    if 'LVTM' not in par:
        continue
    multi = [[], []]
    rawLLTM = PAR[indp]['rawLLTM']
    res, new, conf, sep, corr = fcnAFIL.resnewconsep(rawLLTM)

    # third column (TP2) of all confluencing lesions
    tp2inconf = [item for sublist in [item2[2] for item2 in conf]
                 for item in sublist]
    # third column (TP2) of all separating lesions
    tp2insep = [item for sublist in [item2[2] for item2 in sep]
                for item in sublist]
    # TP2 lesions in multiple confluencing and separating
    multi[1] = [value for value in tp2insep if value in tp2inconf]
    # TP1 lesions in multiple confluencing and separating
    multi[0] = [value[1] for value in conf if
                list_intersection(value[2], multi[1])]
    if not multi[0]:
        continue

    print(indp, multi[0], multi[1])

    for inds, study in enumerate(PAR[indp]['studies']):
        # % Load Lesion masks
        filename = study['lesiones_fname']

        img = sitk.ReadImage(filename)

        array = sitk.GetArrayFromImage(img) != 0

        # % Label lesion masks
        labels, numlabels = label(array)
        lblmulti = np.zeros_like(labels)
        index = np.isin(labels, multi[inds])
        lblmulti[index] = labels[index]

        # sum(sum(sum(labels)))
        # sum(sum(sum(lblmulti)))
        # np.unique(lblmulti)
        # np.unique(labels)

        imgoverlay = sitk.Cast(sitk.GetImageFromArray(lblmulti),
                               sitk.sitkUInt8)
        imgoverlay.CopyInformation(img)
        foldername = PAR[indp]['studiesname']
        os.makedirs(os.path.join(outdatadir, foldername), exist_ok=True)
        filename = "labels_%d_multiconfsep .nii.gz" % (inds+1)
        sitk.WriteImage(imgoverlay,
                        os.path.join(outdatadir, foldername, filename))
        print("    "+os.path.join(outdatadir, foldername, filename))

    # break

# %% Listas por comprension, lambda, filter, map
# creo una lista de diccionarios
a = [{'a': 1, 'b': 'string a'}, {'a': 2, 'b': 'string b'}]

# lista por comprension
[elem for elem in a if elem.get('a') == 1]
[str(elem['a']) + elem['b'] for elem in a if elem.get('a') == 1]

# filter y lambda functions
list(filter(lambda x: x.get('a') == 1, a))
for i in (filter(lambda x: x.get('a') == 1, a)):
    print(i)

# map
list(map(lambda x: x.get('a') == 1, a))

# %% Ejemplo de NiBabel

inputImageFileName = 'E:/REPOS/AFIL/data/P007-0m_FLAIR_brain_to_0m_T1.nii'
img = nib.load(inputImageFileName)
header = img.header
print(header)


# %% Prueba de SimpleElastix (No funca)
print(sitk.Version())
elastixImageFilter = sitk.ElastixImageFilter()

reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName(inputImageFileName)
image = reader.Execute()

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(sitk.ReadImage("fixedImage.nii"))
elastixImageFilter.SetMovingImage(sitk.ReadImage("movingImage.nii"))
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage())


# %% Covertir los archivos a 1mm
for index, st in ST.iterrows():
    st_t1 = st['T1_fname']
    st_flair = st['FLAIR_fname']
    print('Processing ' + '/'.join(st_t1.split('\\')[-4:]))
    print('   Loading T1 and FLAIR images')
    nii_t1 = nib.load(st_t1)
    nii_flair = nib.load(st_flair)

    # sx, sy, sz = nii.header.get_zooms()
    # volume = sx * sy * sz

    print('   Transforming T1 and FLAIR images to 1mm ')
    nii_t1_1mm = processing.conform(nii_t1)
    nii_flair_1mm = processing.conform(nii_flair)

    # nii_t1_1mm_filename = '/'.join(st_t1.split('/')[-5:]).replace(
    #                      'T1', 'T1_1mm')
    # nii_flair_1mm_filename = '/'.join(st_flair.split('/')[-5:]).replace(
    #                      'FLAIR', 'FLAIR_1mm')
    # nii_lesionseg_1mm_filename = '/'.join(st_flair.split('/')[-5:]).replace(
    #                       'FLAIR', 'lesionseg_1mm')

    nii_t1_1mm_filename = st_t1.replace('T1', 'T1_1mm')
    nii_flair_1mm_filename = st_flair.replace('FLAIR', 'FLAIR_1mm')
    # nii_lesionseg_1mm_filename = st_flair.replace('FLAIR', 'lesionseg_1mm')

    nib.save(nii_t1_1mm, nii_t1_1mm_filename)
    nib.save(nii_flair_1mm, nii_flair_1mm_filename)

print('Listo')

# %% ejemplo de Corregistrar

elastixdir = 'E:/Entelai/elastix-5.0.1-win64/tutorial/'
fixfname = os.path.join(elastixdir, 'cbct.mha')
movfname = os.path.join(elastixdir, 'ct.mha')
outdir = os.path.join(elastixdir, 'out/')
paramfname = os.path.join(elastixdir, 'custom_rigid.txt')

# elastix -f fixedImage.ext -m movingImage.ext -out outputDirectory -p parameterFile.txt
cmdlist = [os.path.join(elastixdir, "elastix"), "-f", fixfname, "-m",
           movfname, "-out", outdir, "-p", paramfname]
out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT).stdout.read()
print(out.decode())

# transformix -in ct.mhd -out dirres -tp dir/TransformParameters.0.R1.txt -def all
infname = movfname
tpfname = os.path.join(outdir, 'TransformParameters.0.txt')
cmdlist = [os.path.join(elastixdir, "transformix"), "-in", infname,
           "-out", outdir, "-tp", tpfname]
out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT).stdout.read()
print(out.decode())

# %% ejemplo de Corregistrar con nuestros datos
elastixdir = 'E:/Entelai/elastix-5.0.1-win64/tutorial/'
fixfname = 'E:/Entelai/DATA/VVS/Visita_1/A/T1_Reg_1A.nii.gz'
movfname = 'E:/Entelai/DATA/VVS/Visita_1/A/FLAIR_Reg_1A.nii.gz'
outdir = 'E:/Entelai/elastix-5.0.1-win64/out'
paramfname = os.path.join(elastixdir, 'custom_rigid.txt')

# elastix -f fixedImage.ext -m movingImage.ext -out outputDirectory -p parameterFile.txt
cmdlist = [os.path.join(elastixdir, "elastix"), "-f", fixfname, "-m",
           movfname, "-out", outdir, "-p", paramfname]
out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT).stdout.read()
print(out.decode())

# transformix -in ct.mhd -out dirres -tp dir/TransformParameters.0.R1.txt -def all
infname = movfname
tpfname = os.path.join(outdir, 'TransformParameters.0.txt')
cmdlist = [os.path.join(elastixdir, "transformix"), "-in", infname, "-out",
           outdir, "-tp", tpfname]
out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT).stdout.read()
# genera mhd y raw
print(out.decode())

# convierto a nii.gz
img = sitk.ReadImage(os.path.join(outdir, "result.mhd"))
sitk.WriteImage(img, os.path.join(outdir, "output.nii.gz"))

# cargo el archivo de transformacion y leo la matriz de transformacion
with open(os.path.join(outdir, "TransformParameters.0.txt"), 'r') as fp:
    for line in fp:
        if line.startswith('(TransformParameters '):
            trf = line[21:-2]
            break
trf.split()
[float(i) for i in trf.split()]

# %% corregistrar
elastixdir = 'E:/Entelai/elastix-5.0.1-win64/tutorial/'
outdir = 'E:/Entelai/elastix-5.0.1-win64/out'
paramfname = os.path.join(elastixdir, 'custom_rigid.txt')

subjects = ['VVS', 'FRO', 'PAM', 'CIM', 'TSM', 'GAM']
visitas = ['1', '2']
ABs = ['A', 'B']
seqs = ['T1', 'FLAIR']

subjects = ['VVS']
t0 = time.time()
M = []
for indsuj, subject in enumerate(subjects):
    print(subject)
    m = {}
    for visita in visitas:
        for AB in ABs:
            for seq in seqs:
                # fixfname = 'E:/Entelai/elastix-5.0.1-win64/tutorial/ct.mha'
                # fixfname = 'E:/Entelai/elastix-5.0.1-win64/tutorial/cbct.mha'
                # fixfname = ST['fname_T1_1A'][subject]
                fixfname = 'E:/Entelai/DATA/VVS/Visita_1/A/T1_1mm_Reg_1A.nii.gz'
                movfname = 'E:/Entelai/DATA/VVS/Visita_1/A/translated_T1_1mm_Reg_1A.nii'
                movfname = 'E:/Entelai/DATA/VVS/Visita_1/A/rotated_T1_1mm_Reg_1A.nii'

                # movfname = ST['fname_T1_1A'][subject]
                # fieldname = "fname_%s_%s%s" % (seq,visita, AB)
                # movfname = ST[fieldname][subject]

                # elastix -f fixedImage.ext -m movingImage.ext -out outputDirectory -p parameterFile.txt
                cmdlist = [os.path.join(elastixdir, "elastix"), "-f", fixfname,
                           "-m", movfname, "-out", outdir, "-p", paramfname]
                out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT).stdout.read()

                with open(os.path.join(outdir, "TransformParameters.0.txt"),
                          'r') as fp:
                    for line in fp:
                        if line.startswith('(TransformParameters '):
                            trf = line[21:-2]
                            break
                trf = [float(i) for i in trf.split()]
                fieldname = "tm_%s_%s%s" % (seq, visita, AB)
                m[fieldname] = trf

                infname = movfname
                tpfname = os.path.join(outdir, 'TransformParameters.0.txt')
                cmdlist = [os.path.join(elastixdir, "transformix"), "-in",
                           infname, "-out", outdir, "-tp", tpfname]
                out = subprocess.Popen(cmdlist, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT).stdout.read()

                # convierto a nii.gz
                img = sitk.ReadImage(os.path.join(outdir, "result.mhd"))
                sitk.WriteImage(img, os.path.join(outdir, "output.nii.gz"))

                print('\t%s %2.2f' % (fieldname, time.time() - t0))
                print(trf)
                break
            else:
                continue
            break
        else:
            continue
        break
    else:
        continue
    break
    M.append(m)
print('Listo')

# %%  image overlay according to lession type

subjects = ['VVS', 'FRO', 'PAM', 'CIM', 'TSM', 'GAM']
foldername = './out'
os.makedirs(foldername, exist_ok=True)
tstart = time.time()
for subject in subjects:
    imgT1 = sitk.ReadImage('E:/Entelai/DATA/%s/Visita_1/A/T1_1mm_Reg_1A.nii.gz'
                           % (subject))

    uint8image = sitk.Cast(sitk.RescaleIntensity(imgT1), sitk.sitkUInt8)
    # Invierto la imagen para que el fondo sea blanco
    uint8image_r = sitk.InvertIntensity(
        sitk.IntensityWindowing(uint8image,
                                windowMinimum=15, windowMaximum=200))

    m1 = sitk.ReadImage('relabeled_lesionmask-%s_1_A.nii' % (subject))
    m2 = sitk.ReadImage('relabeled_lesionmask-%s_2_A.nii' % (subject))

    imgm1 = sitk.GetArrayFromImage(m1)
    imgm2 = sitk.GetArrayFromImage(m2)

    um1 = np.unique(imgm1)
    um2 = np.unique(imgm2)

    # labels que estan en ambos
    interseccion = [value for value in um1 if value in um2]
    interseccion.remove(0)
    u1 = [value for value in um1 if value not in um2]  # solo en m1
    u2 = [value for value in um2 if value not in um1]  # solo en m2

    onlyin1 = ismember(imgm1, u1)[0]  # voxels de labels solo en m1
    onlyin2 = ismember(imgm2, u2)[0]  # voxels de labels solo en m2
    inboth = ismember(imgm1, interseccion)[0] | ismember(imgm2,
                                                         interseccion)[0]

    # armo la imagen con todos los labels por color
    # 1: Label ambas: voxel solo en m1 rojo Corresponding Shrink
    # 2: Label ambas: voxel solo en m2 verde Corresponding Grow
    # 3: Label ambas: voxel en ambas amarillo Corresponding Stable
    # 4: Label solo m1 azul Resolving
    # 5: Label solo m2 cyan New
    imgoverlay = onlyin1 * 4 + onlyin2 * 5 + inboth * (
        ((imgm1 > 0) & (imgm2 == 0)) * 1 +
        ((imgm1 == 0) & (imgm2 > 0)) * 2 +
        ((imgm1 > 0) & (imgm2 > 0)) * 3)

    img_overlay = sitk.Cast(sitk.GetImageFromArray(imgoverlay), sitk.sitkUInt8)
    img_overlay.CopyInformation(imgT1)

    #               5            1           2           3            4
    #            cyan           red       green      yellow       blue
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0),
              (255, 255, 0), (0, 0, 255)]
    alpha = 1

    sitk_overlay_img = sitk.LabelOverlay(
        uint8image, img_overlay, opacity=alpha, backgroundValue=0.0,
        colormap=sum(colors, ()))
    sitk_overlay_img_r = sitk.LabelOverlay(
        uint8image_r, img_overlay, opacity=alpha, backgroundValue=0.0,
        colormap=sum(colors, ()))

    sitk.WriteImage(
        img_overlay, os.path.join(foldername,
                                  'lesionseg_colors_%s.nii.gz' % (subject)))
    sitk.WriteImage(
        sitk_overlay_img, os.path.join(foldername, 'T1+lesionseg_%s.nii.gz.dcm'
                                       % (subject)))
    sitk.WriteImage(
        sitk_overlay_img_r, os.path.join(foldername,
                                         'T1+lesionseg_%s_r.nii.gz.dcm'
                                         % (subject)))

    # plt.imshow(sitk.GetArrayFromImage( sitk_overlay_img)[157,:,:])
    # plt.imshow(sitk.GetArrayFromImage( sitk_overlay_img)[:,:,115])
    # plt.imshow(sitk.GetArrayFromImage( sitk_overlay_img)[:,:,110])
    # plt.imshow(sitk.GetArrayFromImage( sitk_overlay_img)[:,:,95])

print(time.time() - tstart)

# %% Etiquetas de color
colors2 = [colors[i] for i in [1, 3, 2, 4, 0]]
fig = plt.figure()
plt.rcParams['figure.facecolor'] = 'black'
for color in colors2:
    plt.bar(1, 1, color=np.array(color) / 255)
plt.legend(('Corresponding. Shrink', 'Corresponding. Stable',
            'Corresponding. Grow', 'Resolving', 'New'))
ax = plt.gca()
ax.set_facecolor('black')
ax.set_position([0, 0, 0, 0])

plt.rcParams['figure.facecolor'] = 'white'



# %% construyo imagenes para el paper

from matplotlib import cm
from matplotlib.colors import ListedColormap

plt.figure(10)

i1 = [0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,1,0]
i2 = [0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0]

plt.subplot(3,1,1)
plotdata = [i1,i2] 
plt.imshow(plotdata,
           cmap = 'binary')
plt.yticks(ticks=[0,1],labels=["Baseline","Follow-up"])
plt.xticks([])
plt.ylim([1.5,-1])
plt.box(False)
plt.ylabel('a',rotation='horizontal')

for i,row in enumerate(plotdata):
    for j,data in enumerate(row):
        if data>0:
            plt.text(j,i,data,fontsize=6,horizontalalignment='center',verticalalignment='center',color=[1,1,1])
        if data==0:
            plt.text(j,i,data,fontsize=6,horizontalalignment='center',verticalalignment='center',color=[0,0,0])            

# cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
#                         'Dark2', 'Set1', 'Set2', 'Set3',
#                         'tab10', 'tab20', 'tab20b', 'tab20c']

mycmap = cm.get_cmap('tab10', 10)
mycmap = mycmap(np.linspace(0, 1, 10))
mycmap = np.vstack([np.array([1,1,1,1]),mycmap])
mycmap = ListedColormap(mycmap)

plt.subplot(3,1,2)
i2b=label(i2)[0]
i2b=i2b+2
i2b[i2b==2]=0
i1b=label(i1)[0]
plotdata = [i1b,i2b]
plt.imshow(plotdata,
           cmap = mycmap)
plt.yticks(ticks=[0,1],labels=["Baseline","Follow-up"])
plt.xticks([])
plt.ylim([1.5,-1])
plt.box(False)
plt.ylabel('b',rotation='horizontal')

for i,row in enumerate(plotdata):
    for j,data in enumerate(row):
        if data>0:
            plt.text(j,i,data,fontsize=6,horizontalalignment='center',verticalalignment='center')


plt.subplot(3,1,3)
plotdata = np.array(label([i1,i2])[0])
plt.imshow(plotdata,
           cmap = mycmap)
plt.yticks(ticks=[0,1],labels=["Baseline","Follow-up"])
plt.xticks(ticks=[1.5,5.5,10,17,30,42],labels=['New','Res','Corr','Sep','Conv','MCS'])
plt.ylim([1.5,-1])
plt.box(False)
plt.ylabel('c',rotation='horizontal')

for i,row in enumerate(plotdata):
    for j,data in enumerate(row):
        if data>0:
            plt.text(j,i,data,fontsize=6,horizontalalignment='center',verticalalignment='center')
            
fig = plt.gcf()
fig.set_size_inches(8, 2)

# %% pruebo de cargar un dicom dcm
fig = plt.figure()
for i in range(20):
    fname = 'E:/Entelai/DATA/THRESH~1/93143B~1/46F15E~1/UNKNOW~1/MR Entelai Segmentacin de Lesiones/MR%06d.dcm'%(i)
    
    img1 = sitk.ReadImage(fname)
    arr1 = sitk.GetArrayFromImage(img1)
    plt.imshow(arr1[0])
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(.000001)