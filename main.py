from PIL import Image
import numpy as np
import math
from scipy.fftpack import dct

#Image as input
image = Image.open("demo.png")
image_rgb = image.convert("RGB")
image_array = np.array(image_rgb)

def display():
    print(image_rgb.size)
    print(image_rgb.mode)

#Step:1 Color Space Conversion and Downsampling
def ColorSpaceConversion(img):
    Y   = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    Cb  = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
    Cr  = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] -0.0813 * img[:,:,2] + 128
    return Y,Cb,Cr

#Input Argument must be a 2D np array
def Downsampling(img):
    x,y = img.shape[:2]
    downsampled = np.zeros((int(x/2),int(y/2)))
    for i in range (0,x-x %2,2):
        for j in range (0,y-y%2,2):
            downsampled[i//2,j//2] = (img[i+0,j+0] + img[i+0,j+1]+ img[i+1,j+0]+ img[i+1,j+1]) / 4 
    return downsampled

Y,Cb,Cr = ColorSpaceConversion(image_array)
Downsampled_Cb = Downsampling(Cb)
Downsampled_Cr = Downsampling(Cr)
img = Image.fromarray(Downsampled_Cb)
img.show()

#Step:2 DCT and quantanization 
DCT  = np.zeros(Y.shape)
x,y = DCT.shape[0:2]
Quantanization_table = [[ 1,  1,  1,  2,  3,  5,  7,  8],
 [ 1,  1,  2,  2,  4,  7,  8,  7],
 [ 1,  2,  2,  3,  5,  8,  9,  7],
 [ 2,  2,  3,  4,  7, 22, 31,  9],
 [ 3,  4,  5,  7,  9, 25, 24, 32],
 [ 5,  7,  8, 32, 23, 24, 26, 22],
 [ 7,  8,  9, 32, 24, 26, 26, 23],
 [ 8,  7,  7,  9, 32, 22, 23, 23]]

#Using scippy fft to perform DCT
def blockDCT(block):
    block = block.astype(float) - 128
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

pad_x = (8 - x % 8) % 8
pad_y = (8 - y % 8) % 8
Y_padded = np.pad(Y, ((0,pad_x),(0,pad_y)), mode='constant', constant_values=0)

DCT_padded = np.zeros_like(Y_padded)
for i in range(0, Y_padded.shape[0], 8):
    for j in range(0, Y_padded.shape[1], 8):
        DCT_padded[i:i+8,j:j+8] = blockDCT(Y_padded[i:i+8,j:j+8]) // Quantanization_table


DCT = DCT_padded[:x,:y]
print(DCT)


image = Image.fromarray(DCT)
image.show()
