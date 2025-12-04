#This script contains the manual calculation of the DCT coeffiencients of 8*8 pixel block
import numpy as np
import math 

def computeDCT(block8_8): 
    block8_8 = block8_8.astype(float) - 128 #Shifted so that the pixel value is centered aroud 0 
    DCT_8_8 = np.zeros((8,8)) #Initialize the Coeffiient matrix for 8*8 cosine frequency block 
    for u in range (0,8): 
        for v in range (0,8): 
            sum_factor = 0 
            alpha_u = 1 if u > 0 else 1/math.sqrt(2) 
            alpha_v = 1 if v > 0 else 1/math.sqrt(2) 
            for x in range (0,8): 
                for y in range (0,8): 
                    sum_factor = sum_factor+block8_8[x,y] * np.cos(((2*x+1)*u*np.pi)/16) * np.cos(((2*y+1)*v*np.pi)/16) 
        DCT_8_8[u,v] = (1/4) * alpha_u * alpha_v * sum_factor 
        return DCT_8_8