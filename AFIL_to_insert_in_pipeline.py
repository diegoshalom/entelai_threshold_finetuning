import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure
from ismember import ismember
import matplotlib.pyplot as plt
import lesion_classifiers as lc
from inspect import signature

def AFIL(img1, img2, param):
    """
    Automatic Follow-up of Individual Lesions (AFIL)
    Calculate and classify lesions
    Parameters
    ----------
    img1, img2: sitkimages of lesions

    Returns
    -------
    imgout1, imgout2: sitk images of segmentations, labeled as:
        1=new
        2=resolving
        3=small
        4=growing
        5=stable
    """
    array1 = sitk.GetArrayFromImage(img1) > 0.5

    array2 = sitk.GetArrayFromImage(img2) > 0.5

    # print("     ",array1.shape,array2.shape)
    matrix4d = np.stack((array1, array2))
    
    # s= generate_binary_structure(4,4)
    # labels, numlabels = label(matrix4d, structure=s)
    
    labels, numlabels = label(matrix4d)
    
    labels1 = labels[0, :, :, :]
    labels2 = labels[1, :, :, :]
    
    vol1, x = calculate_volume_labels(labels1, numlabels)
    vol2, x = calculate_volume_labels(labels2, numlabels)
    LVTM = np.transpose(np.stack([x, vol1, vol2]))
    
    fcn_classify_lesions = param['fcn_classify_lesions']        
    sig = signature(fcn_classify_lesions)
    if len(sig.parameters)==2:
        index = fcn_classify_lesions(vol1, vol2)
    else:
        slope = param['slope']
        thres = param['thres']
        index = fcn_classify_lesions(vol1, vol2, thres, slope)
    
    Imout = [0, 0]    
    for indim, Im in enumerate([labels1, labels2]):
        
        # build array with new=1, res=2, etc
        array = np.zeros_like(Im)
        for lab, j in zip(['new', 'res', 'small', 'grow', 'stable'],
                         [1, 2, 3, 4, 5]):
            array += j * ismember(Im, 1+np.where(index[lab])[0])[0]
            
        imgoverlay = sitk.Cast(sitk.GetImageFromArray(array), sitk.sitkUInt8)
        imgoverlay.CopyInformation(img1)
        Imout[indim] = imgoverlay
                        
    return Imout[0], Imout[1], LVTM, index, labels1, labels2, numlabels

def calculate_volume_labels(labels, maxlabel):
    """    
    Calculate volumes of lesion labels

    Parameters
    ----------
    labels : np.array 3-D
    maxlabel : maximum label number

    Returns
    -------
    y : np array
        volume
    xy : np array
        index

    """
    valores = range(1, maxlabel + 2)
    y, x = np.histogram(labels.flatten(), valores)
    x = np.delete(x, -1)

    return y, x
    
    
