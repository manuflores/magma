import numpy as np 
from scipy.special import gamma

def area_unit_sphere(d):
    num = 2*np.pi**(d/2)
    denom = 0.5*gamma(d/2)
    surf_area =num/denom
    return surf_area

def vol_unit_sphere(d): 
    num= np.pi**(d/2)
    denom= (d/2)*gamma(d/2)
    vol=num/denom
    return vol

