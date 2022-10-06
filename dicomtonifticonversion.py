
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import fcnAFIL
import lesion_classifiers as lc
import time
import SimpleITK as sitk


# datadir = 'E:/Entelai/DATA/threshold_finetuning'
# codedir = 'E:/DIEGO_COMPARTIDO/Dropbox/_LABO/entelai/'
# df = pd.read_excel(os.path.join(codedir,'false_positives_neuro_desm.xlsx'))
# print(df)

# print(df['anon_id'])

# L = []
# for index, row in df.iterrows():
#     anon_id = row['anon_id']
#     patient_id = row['patient_id']
#     study_date = row['study_date']
#     dir = os.path.join(datadir,anon_id)
#     with os.scandir(dir) as entries:
#         for entry in entries:
#             dir = os.path.join(datadir,anon_id,entry.name,'Unknown Study')            
#             l = os.listdir(dir)
            
#             # print(index,anon_id,[t for t in l if t.endswith('_t1_seglesions_procentelailesionseg')])
#             out = [t for t in l if t.endswith('_t1_seglesions_procentelailesionseg')]
#             if len(out)==0:
#                 out = ''
#             else:
#                 out = out[0]

#             L.append({'anon_id':anon_id,'path':dir,'patient_id':patient_id,'study_date':study_date,'lesionseg':out})
# df = pd.DataFrame(L)
# # print(df)
# df.to_csv(os.path.join(codedir,'out.csv'),index=False)

# from entelaiutils import conversion
# for index, row in df.iterrows():
#     if len(row['lesionseg'])>0:    
#         dicom_dir = os.path.join(row['path'],row['lesionseg'])    
#         output_file = os.path.join(datadir,'out','%s_%s.nii.gz'%(row['patient_id'],row['study_date']))
#         conversion.dicom_to_nifti(dicom_dir,output_file)
#         print(index,output_file)

if os.environ['COMPUTERNAME']=='DIEGO-DESKTOP':
    datadir = 'E:/Entelai/DATA/threshold_finetuning/data completa'
    codedir = 'E:/REPOS/entelai_threshold_finetuning/'
else:
    datadir = 'C:/Entelai/DATA/threshold_finetuning/data completa'
    codedir = 'C:/Users/Diego/Dropbox/_LABO/entelai'
    
outdatadir = os.path.join(datadir, 'out')

def build_pairs():
    '''
    PAR: list of pairs: 
        Each pair is a dict:
            studiesname: patient_id
            stidies: (chronologicaly ordered) list of (two) dict studies with 
                        anon_id
                        original_id
                        patient_id
                        study_date
    '''
    df = pd.read_excel(os.path.join(codedir,'false_positives_neuro_desm.xlsx'))
    pids = np.unique(df['patient_id'])
    PAR =[]
    for pid in pids:
        dfuid = df[df['patient_id']==pid].sort_values(by=['study_date'])

        st = []
        for index, row in dfuid.iterrows():
            st.append(dict(row))

        par = {}
        par['studiesname'] = str(pid)
        par['studies'] = st

        PAR.append(par)           

    for i,par in enumerate(PAR):        
        # add lesiones_fname (if exists)
        for j,row in enumerate(par['studies']):
            anon_id = row['anon_id']
            patient_id = row['patient_id']
            study_date = row['study_date']        
            dir = os.path.join(datadir,anon_id,'derivatives','ENTELAI')
            if not os.path.exists(dir):
                lesiones_fname = ""
            else:
                dirname = os.listdir(dir)[0]
                lesiones_fname = os.listdir(os.path.join(dir,dirname))[0]
                lesiones_fname = os.path.join(dir,dirname,lesiones_fname)
            PAR[i]['studies'][j]['lesiones_fname'] = lesiones_fname

        # add flair_fname (if exists)
        anon_id = par['studies'][1]['anon_id'] # flair files are in chronologically last folder
        dir = os.path.join(datadir,anon_id)
        if not os.path.exists(dir):
            flair_fname_0 = ""
            flair_fname_1 = ""
        else:
            dirnames = os.listdir(dir)
            dirname = [x for x in dirnames if x.startswith('sub-adhocrun-')]
            dir = os.path.join(dir,dirname[0])
            files = os.listdir(dir)
            flair_fname_0 = [x for x in files if x.endswith('flair_coreg-t1-prev_1mm.nii.gz')][0]
            flair_fname_0 = os.path.join(dir,flair_fname_0)        
            flair_fname_1 = [x for x in files if x.endswith('flair_coreg-t1_1mm.nii.gz')][0]
            flair_fname_1 = os.path.join(dir,flair_fname_1)        
        PAR[i]['studies'][0]['FLAIR_fname'] = flair_fname_0
        PAR[i]['studies'][1]['FLAIR_fname'] = flair_fname_1

    # Add comment
    df = pd.read_excel(os.path.join(codedir,'false_positives_neuro_desm.xlsx'),sheet_name="Comentarios")
    for i,par in enumerate(PAR):                
        comment = df[df['patient_id']==int(par['studiesname'])]['comment'].values[0]
        PAR[i]['comment'] = comment

    return PAR

PAR = build_pairs()
       


param = {"thres":np.nan, "slope":np.nan, "growtype":"largesmall_01p", \
    'fcn_classify_lesions':lc._classify_lesions_6SD,   \
        'condname':'largesmall_01p_new10'}

t = time.time()
for i, par in enumerate(PAR):
    print('Inicio %s'%(par['studiesname']))

    if len(par['studies'][0]['lesiones_fname'])==0:
        print('Empty filename')
        continue

    # if par['studiesname'] == "462859":
    #     print("Salteo estudio porque los tamaños no corresponden")
    #     continue
    # print(par['studies'][0]['patient_id'],par['studies'][1]['patient_id'])
    # print(par['studies'][0]['anon_id'],par['studies'][1]['anon_id'])
    # print(par['studies'][0]['lesiones_fname'],par['studies'][1]['lesiones_fname'])    
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
    