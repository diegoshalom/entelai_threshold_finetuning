import numpy as np

def _classify_lesions_6tau_samereso(v1, v2):
    """
    Classify lesions in grow/new and stable, according to vol1 and vol2
    Parameters
    ----------
    v1, v2: lists of volumes of lesions an timepoints 1 and 2

    Returns
    -------
    index: a dictionary of classes of lesions. Each one is a list of boolean:
        -grow: Growing/New
        -stable: Stable
    """    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1==0) & (v2>30) #always false
    indexcor = ~(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & ((v1<65) & (v2>v1*1.1554+30) |
                            (v1>=65) & (v2>v1*1.61692) |
                            ((v1==0) & (v2>=30)))
    indexstable = indexcor & ~(indexgrow)

    index ={'grow': indexgrow, 'stable': indexstable, 'new': indexnew}

    return index    

def _classify_lesions_6tau_diffreso(v1, v2):
    """
    Classify lesions in grow/new and stable, according to vol1 and vol2
    Parameters
    ----------
    v1, v2: lists of volumes of lesions an timepoints 1 and 2

    Returns
    -------
    index: a dictionary of classes of lesions. Each one is a list of boolean:
        -grow: Growing/New
        -stable: Stable
    """    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1==0) & (v2>30) #always false
    indexcor = ~(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & ((v1<77) & (v2>v1*2.22703+30) |
                            (v1>=77) & (v2>v1*2.61664) |
                            ((v1==0) & (v2>=30)))
    indexstable = indexcor & ~(indexgrow)

    index ={'grow': indexgrow, 'stable': indexstable, 'new': indexnew}
    
    return index

def _classify_lesions_6SD(v1, v2, thres, slope):
    """
    Classify lesions in grow/new and stable, according to vol1 and vol2
    Parameters
    ----------
    v1, v2: lists of volumes of lesions an timepoints 1 and 2

    Returns
    -------
    index: a dictionary of classes of lesions. Each one is a list of boolean:
        -grow: Growing/New
        -stable: Stable
    """    
    thresnew = 30
    slopehigh = 1+.10282*6
    slopelow =  1.1554 # La pendiente de la recta que pasa por (0,30) y (65,slopehigh*65) 

    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew =  (v1 == 0) & (v2 >= thresnew)  
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*slopelow+thresnew) | np.bitwise_and(v1>=65, v2>v1*slopehigh) )
    indexstable = indexcor & np.bitwise_not(indexgrow)

    # index ={'grow': indexgrow, 'stable': indexstable}
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}    
    return index

def _classify_lesions_4tau_samereso(v1, v2):
    """
    Classify lesions in grow/new and stable, according to vol1 and vol2
    Parameters
    ----------
    v1, v2: lists of volumes of lesions an timepoints 1 and 2

    Returns
    -------
    index: a dictionary of classes of lesions. Each one is a list of boolean:
        -grow: Growing/New
        -stable: Stable
    """    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.02104+25.365) | np.bitwise_and(v1>=65, v2>v1*1.41127) | ((v1==0) & (v2 > 17.6)))
    indexstable = indexcor & np.bitwise_not(indexgrow)

    # index ={'grow': indexgrow, 'stable': indexstable}
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}    
    return index

def _classify_lesions_4tau(v1, v2, thres, slope):
    """
    Classify lesions in grow/new and stable, according to vol1 and vol2
    Parameters
    ----------
    v1, v2: lists of volumes of lesions an timepoints 1 and 2

    Returns
    -------
    index: a dictionary of classes of lesions. Each one is a list of boolean:
        -grow: Growing/New
        -stable: Stable
    """    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.02104+25.365) | np.bitwise_and(v1>=65, v2>v1*1.41127) | ((v1==0) & (v2 > 17.6)))
    indexstable = indexcor & np.bitwise_not(indexgrow)

    # index ={'grow': indexgrow, 'stable': indexstable}
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}    
    return index

def classify_lesion_largesmall_4sigma(v1, v2, thres, slope):
    # Para 4 SD
    # Ecuación de la recta 1: V_2C = 25.365 + V1 * 1.02104
    # Ecuación de la recta 2: V_2C = V1 * 1.41127
    # 4*SD = 17.6 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.02104+25.365) | np.bitwise_and(v1>=65, v2>v1*1.41127))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_largesmall_4sigma_new10(v1, v2, thres, slope):
    # Para 4 SD
    # Ecuación de la recta 1: V_2C = 25.365 + V1 * 1.02104
    # Ecuación de la recta 2: V_2C = V1 * 1.41127
    # 4*SD = 17.6 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.02104+25.365) | np.bitwise_and(v1>=65, v2>v1*1.41127) | ((v1==0) & (v2 > 17.6)))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_largesmall_01p(v1, v2, thres, slope):
    # Para 0.1%FPR (3 SD)
    # Ecuación de la recta 1: V_2C = 19 + V1 * 1.01615
    # Ecuación de la recta 2: V_2C = V1 * 1.30845
    # 3*SD = 13.2 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.01615+19) | np.bitwise_and(v1>=65, v2>v1*1.30845))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_largesmall_01p_new10(v1, v2, thres, slope):
    # Para 0.1%FPR (3 SD)
    # Ecuación de la recta 1: V_2C = 19 + V1 * 1.01615
    # Ecuación de la recta 2: V_2C = V1 * 1.30845
    # 3*SD = 13.2 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.01615+19) | np.bitwise_and(v1>=65, v2>v1*1.30845) | ((v1==0) & (v2 > 13)))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_largesmall_1p(v1, v2, thres, slope):
    # Para 1%FPR (2.32SD)
    # Ecuación de la recta 1: V_2C = 15 + V1 * 1.00843
    # Ecuación de la recta 2: V_2C = V1 * 1.2392    
    # 2.32*SD = 10.235 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.00843+15) | np.bitwise_and(v1>=65, v2>v1*1.2392))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_largesmall_1p_new10(v1, v2, thres, slope):
    # Para 1%FPR (2.32SD)
    # Ecuación de la recta 1: V_2C = 15 + V1 * 1.00843
    # Ecuación de la recta 2: V_2C = V1 * 1.2392    
    # 2.32*SD = 10.235 mm3
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (np.bitwise_and(v1<65, v2>v1*1.00843+15) | np.bitwise_and(v1>=65, v2>v1*1.2392) | ((v1==0) & (v2 > 10)))
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index





def classify_lesion_nonew(v1, v2, thres, slope):
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (v2 > thres + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesion_nonew_new10(v1, v2, thres, slope):
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 < 0) #always false 
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = (indexcor & (v2 > thres + v1 + slope*v1) ) | ( (v1 == 0) & (v2 > 10) )
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index
              
def classify_lesion_new(v1, v2, thres, slope):
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 == 0) & (v2 >= thres)    
    indexcor = np.bitwise_not(indexnew)    
    indexgrow = indexcor & (v2 > thres + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)

    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_old(v1, v2):
    umbral = 10
    slope = 0.05
    stdsmall = 9    

    indexres = ((v2 == 0) & (v1 >= umbral)) | ( (v2 > 0) & (v2 < umbral) & (v1 > v2+2*stdsmall))    
    indexnew = ((v1 == 0) & (v2 >= umbral)) | ( (v1 > 0) & (v1 < umbral) & (v2 > v1+2*stdsmall))
    indexsmall = (v1 + v2) < umbral    
    indexcor = np.bitwise_not(indexres | indexnew | indexsmall)    
    indexgrow = indexcor & (v2 > umbral + v1 + slope*v1)            
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_5p_new_vol(v1, v2):
    umbral = 10
    slope = 0.05
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 == 0) & (v2 >= umbral)    
    indexcor = np.bitwise_not(indexnew)    
    indexgrow = indexcor & (v2 > umbral + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_5p_new_sum(v1, v2):
    umbral = 15
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 == 0) & (v2 >= umbral)    
    indexcor = np.bitwise_not(indexnew)    
    indexgrow = indexcor & (v2 > v1 + umbral)
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_5p_nonew_vol(v1, v2):
    umbral = 10
    slope = 0.05
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = (v1 > -1) #always true
    indexgrow = indexcor & (v2 > umbral + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_5p_nonew_sum(v1, v2):
    umbral = 15
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 < 0) #always false
    indexcor = (v1 > -1) #always true
    indexgrow = indexcor & (v2 > v1 + umbral)
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_1p_new_vol(v1, v2):
    umbral = 10
    slope = 0.448
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 == 0) & (v2 >= umbral)    
    indexcor = np.bitwise_not(indexnew)    
    indexgrow = indexcor & (v2 > umbral + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_1p_new_sum(v1, v2):
    umbral = 48
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 == 0) & (v2 >= umbral)    
    indexcor = np.bitwise_not(indexnew)    
    indexgrow = indexcor & (v2 > v1 + umbral)
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_1p_nonew_vol(v1, v2):
    umbral = 10
    slope = 0.448
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 <  0) #always false
    indexcor = (v1 > -1) #always true
    indexgrow = indexcor & (v2 > umbral + v1 + slope*v1)        
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index

def classify_lesions_1p_nonew_sum(v1, v2):
    umbral = 48
    
    indexres = (v1 < 0) #always false
    indexsmall = (v1 < 0) #always false
    indexnew = (v1 < 0) #always false
    indexcor = (v1 > -1) #always true
    indexgrow = indexcor & (v2 > v1 + umbral)
    indexstable = indexcor & np.bitwise_not(indexgrow)
    
    index ={'res': indexres, 'new': indexnew, 'small': indexsmall,
              'grow': indexgrow, 'stable': indexstable}
    return index
