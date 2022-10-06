import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from ismember import ismember
from scipy.ndimage import label
import matplotlib.pyplot as plt

from AFIL_to_insert_in_pipeline import AFIL
from AFIL_to_insert_in_pipeline import calculate_volume_labels

# import nibabel as nib
# import nibabel.processing as processing
# import subprocess
# import time


def build_PAR_20(datadir):
    
    xls = pd.ExcelFile(os.path.join(datadir,'listado 20.xlsx'))
    ST = pd.read_excel(xls, 'Hoja1')
    xls.close()    
    
    PAR = []
    
    for i in ST.index:
        par = {}        
        par['studiesname'] = "%03d_%s" % (ST.iloc[i]['ID'], ST.iloc[i]['name'])
        st1 = {}
        st2 = {}
        
        folder = ST['folder1'][i]
        st1['T1_fname'] = os.path.join(datadir, folder, 'rT1-conformed.nii.gz')
        st1['FLAIR_fname'] = os.path.join(datadir, folder, 'FLAIR-conformed.nii.gz')
        st1['lesiones_fname'] = os.path.join(datadir, folder, 'lesionseg.nii.gz')


        folder = ST['folder2'][i]
        st2['T1_fname'] = os.path.join(datadir, folder, 'rT1-conformed-cor01.nii.gz')
        st2['FLAIR_fname'] = os.path.join(datadir, folder, 'FLAIR-conformed-cor01.nii.gz')
        st2['lesiones_fname'] = os.path.join(datadir, folder, 'lesionseg-cor01.nii.gz')

        par['studies'] = [st1, st2]
    
        PAR.append(par)
    return PAR


def build_structure_ST(datadir):
    """
    Build a list of dictionaries with info of each study (Image)
    Parameters
    ----------
    datadir: string

    Returns
    -------
    ST : list of dictionaries
        DESCRIPTION.

    """

    xls = pd.ExcelFile('Protocolo 50-18 FLENI Entelai.xlsx')
    ST = pd.read_excel(xls, 'estudios')
    xls.close()
    T1_fname = []
    FLAIR_fname = []
    lesiones_fname = []
    for i in ST.index:
        subject = ST['subject'][i]
        visita = ST['visita'][i]
        AB = ST['AB'][i]
        fname = os.path.join(datadir, 'DATA', subject, 'Visita_' + str(visita),
                             AB, 'T1_1mm_Reg_1A.nii.gz')
        T1_fname.append(fname)
        fname = os.path.join(datadir, 'DATA', subject, 'Visita_' + str(visita),
                             AB, 'FLAIR_1mm_Reg_1A.nii.gz')
        FLAIR_fname.append(fname)
        fname = os.path.join(datadir, 'lesiones', subject,
                             'Visita_' + str(visita), AB,
                             'lesionseg_1mm_Reg_1A.nii.gz')
        lesiones_fname.append(fname)

    ST['T1_fname'] = T1_fname
    ST['FLAIR_fname'] = FLAIR_fname
    ST['lesiones_fname'] = lesiones_fname
    return ST


def build_pairs_of_studies(ST):
    """
    Builds a list of dictionaries with all possible combinations of studies
    Parameters
    ----------
    ST : list of dictionaries
        DESCRIPTION.

    Returns
    -------
    PAR : list of dictionaries
        DESCRIPTION.
        grupo:  0 ≠suj
                1 =suj, =reso, =visit
                2 =suj, =reso, ≠visit
                3 =suj, ≠reso, =visit
                4 =suj, ≠reso, ≠visit
        grupo = ["≠suj", "=suj, =reso, =visit", "=suj, =reso, ≠visit",
                 "=suj, ≠reso, =visit", "=suj, ≠reso, ≠visit"]
    """
    #  %% Armo pares de estudios por categoria
    PAR = []
    for i in range(24):
        for j in range(i + 1, 24):
            par = {}
            st1 = dict(ST.loc[i])
            st2 = dict(ST.loc[j])
            par['studies'] = [st1, st2]
            par['samesubj'] = st1['subject'] == st2['subject']
            par['samereso'] = st1['reso'] == st2['reso']
            par['samevisit'] = st1['visita'] == st2['visita']
            if (par['samesubj'] == False):
                grupo = 0
                subj = "%s_%s" % (st1['subject'], st1['subject'])
            elif (par['samereso'] == True) and (par['samevisit'] == True):
                grupo = 1
                subj = st1['subject']
            elif (par['samereso'] == True) and (par['samevisit'] == False):
                grupo = 2
                subj = st1['subject']
            elif (par['samereso'] == False) and (par['samevisit'] == True):
                grupo = 3
                subj = st1['subject']
            elif (par['samereso'] == False) and (par['samevisit'] == False):
                grupo = 4
                subj = st1['subject']

            par['grupo'] = grupo
            par['subject'] = subj
            reso = ' samereso' if par['samereso'] else ''

            par['studiesname'] = "%s_%d%s vs %s_%d%s%s" % (
                st1['subject'], st1['visita'], st1['AB'],
                st2['subject'], st2['visita'], st2['AB'], reso
                )

            PAR.append(par)
    return PAR


def full_process_pair(par, outdatadir, param):
    filename1 =  par['studies'][0]['lesiones_fname']
    filename2 =  par['studies'][1]['lesiones_fname']
    img1 = sitk.ReadImage(filename1)
    img2 = sitk.ReadImage(filename2)
    
    imout1, imout2, LVTM, index, labels1, labels2, numlabels = AFIL(img1, img2, param)
    
    par['index'] = index
    par['studies'][0]['img'] = img1
    par['studies'][1]['img'] = img2
    par['studies'][0]['labels'] = labels1
    par['studies'][1]['labels'] = labels2
    par['numlabels'] = numlabels
    par['LVTM'] = LVTM
        
    # Copy original FLAIR and lesions files
    copy_flair_lesions(par=par, outdatadir=outdatadir)

    # Export segmentations of labeled lesions
    export_labels(par=par, outdatadir=outdatadir)

    # # Export segmentations with new, res, grow, stable, small
    export_new_resolving(par=par, outdatadir=outdatadir, param=param)

    # Export figure of vol2 vs vol1
    export_vol_vs_vol(par=par, outdatadir=outdatadir, param=param)

    # Export LVTM
    export_LVTM(par=par, outdatadir=outdatadir, param=param)
    
    return par

def analyze_roll_shift(par, roll, param):
    filename1 =  par['studies'][0]['lesiones_fname']
    filename2 =  par['studies'][1]['lesiones_fname']
    img1 = sitk.ReadImage(filename1)
    img2old = sitk.ReadImage(filename2)
    
    arr2 = sitk.GetArrayFromImage(img2old)
    
    arr2=np.roll(arr2,roll[0],axis=0)
    arr2=np.roll(arr2,roll[1],axis=1)
    arr2=np.roll(arr2,roll[2],axis=2)
    
    img2 = sitk.GetImageFromArray(arr2)
    img2.CopyInformation(img2old)
    
    imout1, imout2, LVTM, index, labels1, labels2, numlabels = AFIL(img1, img2, param)
    
    return LVTM, index
    
def small_lesion_filter(maxsize, data):
    """
    Filters out lesions smaller or equal to maxsize
    Parameters
    ----------
    maxsize : int
        DESCRIPTION.
    DATA : list of dict
        DESCRIPTION.

    Returns
    -------
    DATA : list of dict with modified image
        DESCRIPTION.

    """
    labels = data['labels']
    numlabels = data['numlabels']
    volumen = data['volumen']
    for indlabel in range(numlabels):
        if volumen[indlabel] <= maxsize:
            data['array'][labels == indlabel + 1] = False
    return data


def export_new_resolving(par, outdatadir, param):
    fcn_classify_lesions = param['fcn_classify_lesions']
    condname = param['condname']
    v1 = par['LVTM'][:, 1]
    v2 = par['LVTM'][:, 2]
    slope = param['slope']
    thres = param['thres']
    index = fcn_classify_lesions(v1, v2, thres, slope)
    
    # index = par['index']
    indexres = index['res']
    indexnew = index['new']
    indexsmall = index['small']
    # indexcor =  index['cor']
    indexgrow =  index['grow']
    indexstable =  index['stable']    
    
    n1, n2, n3, n4, n5 = [sum(indexnew), sum(indexres), sum(indexsmall),
                          sum(indexgrow), sum(indexstable)]

    for inds, study in enumerate(par['studies']):
        Im = study['labels']    

        arrnew = ismember(Im, 1+np.where(indexnew)[0])[0]
        arrres = ismember(Im, 1+np.where(indexres)[0])[0]
        arrsmall = ismember(Im, 1+np.where(indexsmall)[0])[0]
        arrgrow = ismember(Im, 1+np.where(indexgrow)[0])[0]
        arrstable = ismember(Im, 1+np.where(indexstable)[0])[0]
    
        array = arrnew * 1 + arrres * 2 + arrsmall * 3 + arrgrow * 4 + arrstable * 5
    
        imgoverlay = sitk.Cast(sitk.GetImageFromArray(array), sitk.sitkUInt8)
        # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#ac810e1a6653f07715ff0583ea16d6d2e
        imgoverlay.CopyInformation(par['studies'][1]['img'])
    
        # # INTENTO con LabelOverlay
        # # #               5            1           2           3            4
        # # #            cyan           red       green      yellow       blue
        # colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0),
        #           (255, 255, 0), (0, 0, 255)]
        # alpha = 1
        # emptyarray = array * 0
        # baseimage = sitk.Cast(sitk.GetImageFromArray(emptyarray), sitk.sitkUInt8)
        # baseimage.CopyInformation(par['studies'][1]['img'])
        # imgoverlay = sitk.LabelOverlay(
        #     baseimage, imgoverlay, opacity=alpha, backgroundValue=0.0,
        #     colormap=sum(colors, ()))
    
        # # INTENTO con LabelToRGB
        # imgoverlay = sitk.LabelToRGB(imgoverlay, backgroundValue=0.0,
        #     colormap=sum(colors, ()))
    
        foldername = par['studiesname']
        os.makedirs(os.path.join(outdatadir, foldername, condname), exist_ok=True)        
        filename = "labels %d (n%d, r%d, sm%d, g%d, st%d).nii.gz" % (inds+1, n1, n2, n3, n4, n5)
        sitk.WriteImage(imgoverlay, os.path.join(outdatadir, foldername, condname, filename))


def copy_flair_lesions(par, outdatadir):
    foldername = par['studiesname']
    os.makedirs(os.path.join(outdatadir, foldername), exist_ok=True)
        
    mystr = 'copy "%s" "%s"' % (par['studies'][0]['FLAIR_fname'],
                                os.path.join(outdatadir, foldername,
                                             'FLAIR_1.nii.gz'))
    os.system(mystr)
    mystr = 'copy "%s" "%s"' % (par['studies'][1]['FLAIR_fname'],
                                os.path.join(outdatadir, foldername,
                                             'FLAIR_2.nii.gz'))
    os.system(mystr)

    mystr = 'copy "%s" "%s"' % (par['studies'][0]['lesiones_fname'],
                                os.path.join(outdatadir, foldername,
                                             'lesiones_1.nii.gz'))
    os.system(mystr)
    mystr = 'copy "%s" "%s"' % (par['studies'][1]['lesiones_fname'],
                                os.path.join(outdatadir, foldername,
                                             'lesiones_2.nii.gz'))
    os.system(mystr)


def figure_vol_vs_vol(v1, v2, param,log=False):
    umbral = param['thres']
    slope = param['slope']
    growtype = param['growtype']
    fcn_classify_lesions = param['fcn_classify_lesions']

    lw = 0.3
    maxx = max(max(v1),max(v2))*1.05
    maxy = max(max(v1),max(v2))*1.05
    
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim((0.1,10000))
    # plt.ylim((0.1,10000))
    
    # plt.xlim([-1, 51])
    # plt.ylim([-1, 51])
    # maxx  = plt.xlim()[1]
    # maxy  = plt.ylim()[1]
    # plt.xlim([-1, maxx])
    # plt.ylim([-1, maxy])
    
    # stdsmall = 9
    # plt.plot([0, 0], [umbral, maxy], color='r', label="New", linewidth=lw)
    # plt.plot([0, umbral], [2*stdsmall, 2*stdsmall+umbral], color='r', linewidth=lw)
    # plt.plot([umbral, umbral], [2*stdsmall+umbral, maxy], color='r', linewidth=lw)
    # plt.plot([0, umbral], [maxy, maxy], color='r', linewidth=lw)
    
    # plt.plot([umbral, maxx], [0, 0], color='g', label="Resolving", linewidth=lw)
    # plt.plot([2*stdsmall, 2*stdsmall+umbral], [0, umbral], color='g', linewidth=lw)
    # plt.plot([2*stdsmall+umbral, maxx], [umbral, umbral], color='g', linewidth=lw)
    # plt.plot([maxx, maxx], [0, umbral], color='g', linewidth=lw)
    
    # plt.plot([0, umbral], [0, 0], color='m', label="Small", linewidth=lw)
    # plt.plot([0, 0], [0, umbral], color='m', linewidth=lw)
    # plt.plot([umbral, 0], [0, umbral], color='m', linewidth=lw)
        
    x = np.linspace(0,maxx,201)
    if growtype=='vol':
        plt.plot(x, x + umbral + slope*x, 'b-', label='~Vol', linewidth=lw)        
    elif growtype=='const':
        plt.plot(x, x + umbral, 'b-', label='+Const', linewidth=lw)            
    elif growtype=="largesmall_1p":
        # Ecuación de la recta 1: V_2C = 15 + V1 * 1.00843
        # Ecuación de la recta 2: V_2C = V1 * 1.2392   
        oo= 15
        pend=1.00843
        x = np.linspace (0,65,66)
        y = oo + pend*x
        plt.plot(x, y, 'b-', label='1%% FPR', linewidth=lw)        
        x = np.linspace (65,maxx,100)
        pend=1.2392
        y = pend*x
        plt.plot(x, y, 'b-', linewidth=lw)                
    elif growtype=="largesmall_01p":
        # Ecuación de la recta 1: V_2C = 19 + V1 * 1.01615
        # Ecuación de la recta 2: V_2C = V1 * 1.30845
        #indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.02104+25.365) | np.bitwise_and(v1>=65, v2>v1*1.41127) | ((v1==0) & (v2 > 17.6)))        
        oo= 25.365
        pend=1.02104
        x = np.linspace (0,65,66)
        y = oo + pend*x
        plt.plot(x, y, 'c-', label='0.1%% FPR', linewidth=lw)        
        x = np.linspace (65,maxx,100)
        pend=1.41127
        y = pend*x
        plt.plot(x, y, 'c-', linewidth=lw)

        oo= 30
        pend = 1.1554
        x = np.linspace (0,65,66)
        y = oo + pend*x
        plt.plot(x, y, 'b-', label='0.1%% FPR', linewidth=lw)        
        x = np.linspace (65,maxx,100)
        pend =  1+.10282*6
        y = pend*x
        plt.plot(x, y, 'b-', linewidth=lw)        

    # plt.plot(x, x + pend2*x**(2/3), 'y-', label='~Sup', linewidth=lw)
    # plt.plot(x, x, 'k--', linewidth=lw)    
    plt.plot([0.01,maxx], [0.01,maxx], 'k--', linewidth=lw)    

    slope = param['slope']
    thres = param['thres']
    index = fcn_classify_lesions(v1, v2, thres, slope)    
    indexres = index['res']
    indexnew = index['new']
    indexsmall = index['small']
    # indexcor =  index['cor']
    indexgrow =  index['grow']
    indexstable =  index['stable']
        
    ms = 2
    jitter = lambda x: 0.0*(np.random.rand(sum(x))-.5)
    if log==True:
        v1=np.array(v1,np.float)
        v2=np.array(v2,np.float)
        v1[v1==0] = 0.1
        v2[v2==0] = 0.1
    plt.plot(v1[indexnew]+jitter(indexnew), v2[indexnew]+jitter(indexnew), '.r', markersize=ms)
    plt.plot(v1[indexres]+jitter(indexres), v2[indexres]+jitter(indexres), '.g', markersize=ms)
    plt.plot(v1[indexsmall]+jitter(indexsmall), v2[indexsmall]+jitter(indexsmall), '.m', markersize=ms)
    # plt.plot(v1[indexcor]+jitter(indexcor), v2[indexcor]+jitter(indexcor), '.b', markersize=ms)
    plt.plot(v1[indexstable]+jitter(indexstable), v2[indexstable]+jitter(indexstable), '.b', markersize=ms)
    plt.plot(v1[indexgrow]+jitter(indexgrow), v2[indexgrow]+jitter(indexgrow), '.y', markersize=ms)
    
    
    plt.grid(True)
    plt.axis('square')
    plt.xlabel('Lesion size pre [mm$^3$]')
    plt.ylabel('Lesion size post [mm$^3$]')
    plt.xlim([-1, maxx])
    plt.ylim([-1, maxy])
    # plt.legend(bbox_to_anchor = (1.45, 0.7))
    if log==True:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([.05, maxx])
        plt.ylim([.05, maxy])

def export_vol_vs_vol(par, outdatadir, param):
    plt.figure(figsize=(10, 5))
    plt.clf()   
    vol1 = np.array([elem[1] for elem in par['LVTM']])
    vol2 = np.array([elem[2] for elem in par['LVTM']])
    plt.subplot(1,2,1)
    figure_vol_vs_vol(vol1, vol2, param, log=False)
    title = par['studiesname'] + " '" + par['comment'] + "'"
    plt.title(title)
    plt.subplot(1,2,2)
    figure_vol_vs_vol(vol1, vol2, param, log=True)
    title = par['studiesname'] + " '" + par['comment'] + "'"
    plt.title(title)


    foldername = par['studiesname']
    condname = param['condname']
    os.makedirs(os.path.join(outdatadir, foldername, condname), exist_ok=True)
    filename = "fig_lesions_vol_vs_vol.png"
    plt.savefig(os.path.join(outdatadir, foldername, condname, filename), dpi=150)

    filename = title + ".png"
    os.makedirs(os.path.join(outdatadir, 'imagenes'), exist_ok=True)
    plt.savefig(os.path.join(outdatadir, 'imagenes', filename), dpi=150)


def export_labels(par, outdatadir):    
    for inds, study in enumerate(par['studies']):
        labels = study['labels']   
        numlabels = len(np.unique(labels))-1
        imgoverlay = sitk.Cast(sitk.GetImageFromArray(labels), sitk.sitkUInt8)
        imgoverlay.CopyInformation(study['img'])

        foldername = par['studiesname']
        os.makedirs(os.path.join(outdatadir, foldername), exist_ok=True)
        filename = "labels_%d_(total=%d).nii.gz" % (inds+1, numlabels)
        sitk.WriteImage(imgoverlay, os.path.join(outdatadir, foldername,
                                                 filename))


def export_LVTM(par, outdatadir, param):        
    LVTM = list(par['LVTM'])
    index = par['index']
    categ = index['new']*1+index['res']*2+index['small']*3+index['grow']*4+index['stable']*5
    categs = ('error','new', 'res', 'small', 'grow', 'stable')    
    
    for i,lv in enumerate(LVTM):
        temp =list(lv)
        temp.append(temp[2]-temp[1])
        temp.append(categs[categ[i]])
        LVTM[i] = temp
    
    df = pd.DataFrame(LVTM, columns=['id', 'v1', 'v2', 'dif', 'cat'])
    condname = param['condname']
    fname = os.path.join(outdatadir, par['studiesname'], condname, 'volumes.xlsx')
    df.to_excel(fname, index=False)
    
    # fname = os.path.join(outdatadir, par['studiesname'], 'volumes.csv')
    # np.savetxt(fname, LVTM, delimiter=',', fmt='%d', header='id,v1,v2')


def list_intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def resnewconsep(rawLLTM):
    """
    Given rawLLTM, returns resolving, new, confluencing, separating and
        corresponding lesions

    Parameters
    ----------
    rawLLTM : list

    Returns
    -------
    resolving, new, confluencing, separating, corresponding as lists
    """

    interseccion = [elem[2] for elem in rawLLTM if elem[1] is not np.nan]

    new = [elem for elem in rawLLTM if elem[1] is np.nan]
    resolving = [elem for elem in rawLLTM if len(elem[2]) == 0]
    separating = [elem for elem in rawLLTM if len(elem[2]) > 1]
    # [[50, 50, [42, 49]]]

    # los que estan en la interseccion, flattened
    lll = [item for sublist in interseccion for item in sublist]
    dupes = list(set([x for x in lll if lll.count(x) > 1]))
    confluencing = [elem for elem in rawLLTM if len(elem[2]) > 0 if
                    list_intersection(elem[2], dupes)]
    # ojo que confluencing devuelve 2 cosas, y separating devueve una
    #  [[44, 44, [36]], [55, 55, [36]]]

    corresponding = [[elem[0], elem[1], elem[2][0]] for elem in rawLLTM if
                     elem[1] is not np.nan and len(elem[2]) == 1]

    return resolving, new, confluencing, separating, corresponding

