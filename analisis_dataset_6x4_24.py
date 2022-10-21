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
    datadir = 'E:/Entelai/DATA/Dataset 6x24'
    codedir = 'E:/REPOS/entelai_threshold_finetuning/'
else:
    datadir = 'C:/Entelai/DATA/Dataset_6x4_24'
    codedir = 'C:/REPOS/entelai_threshold_finetuning/'
    
outdatadir = os.path.join(datadir, 'out')

# %% Armo y proceso pares y guardo file
param = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", \
    'fcn_classify_lesions':lc.classify_lesion_largesmall_01p_new10, \
        'condname':'largesmall_01p_new10'}

ST = fcnAFIL.build_structure_ST_6x4(datadir)
PAR = fcnAFIL.build_pairs_of_studies_6x4(ST)    

# PAR = [par for par in PAR if par['samereso']==True and par['samesubj']==True]        
PAR = [par for par in PAR if par['samesubj']==True]        

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
np.savez_compressed(os.path.join(datadir,'PARES_6x4.npz'), PAR=PAR)
# loaded = np.load(os.path.join(datadir,'PARES_6x4.npz'), allow_pickle=True)
# loaded.files
# PAR = list(loaded['PAR'])

# %% cargo file
loaded = np.load(os.path.join(datadir,'PARES_6x4.npz'), allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])
print(len(PAR))
# %% Defino funciones de ploteo

def extract_vp_dvp(PAR,samereso):
    vol1 = np.array([])
    vol2 = np.array([])
    # samereso = [False]
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
# %% exponencial stdev of %Vc vs mean volume

loaded = np.load(os.path.join(datadir,'PARES_6x4.npz'), allow_pickle=True)
loaded.files
PAR = list(loaded['PAR'])

nstd = 2.32634
nstd = 3
nstd = 6

samereso = [True]
samereso = [False]
# samereso = [True, False]

vp, dvp, vol1, vol2 = extract_vp_dvp(PAR,samereso)

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
# plt.plot(x,y-func(x, *popt),'.',label='Residuos')
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
    oo = 30
    
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


# %%
