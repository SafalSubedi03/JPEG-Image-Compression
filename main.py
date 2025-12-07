from PIL import Image
import numpy as np
import math
from scipy.fftpack import dct
from scipy.fftpack import idct
import matplotlib.pyplot as plt

Quantization_table = \
[[1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1]]




Quantization_table = [
 [1, 1, 1, 1, 2, 2, 3, 3],
 [1, 1, 1, 2, 2, 3, 3, 3],
 [1, 1, 2, 2, 3, 3, 3, 3],
 [1, 2, 2, 3, 3, 4, 4, 3],
 [2, 2, 3, 3, 3, 4, 4, 4],
 [2, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 3, 4, 4, 4, 4]
]


# Quantanization_table = [
#  [ 8,  8,  8, 16, 24, 40, 56, 64],
#  [ 8,  8, 16, 16, 32, 56, 64, 56],
#  [ 8, 16, 16, 24, 40, 64, 72, 56],
#  [16, 16, 24, 32, 56, 176, 248, 72],
#  [24, 32, 40, 56, 72, 200, 192, 256],
#  [40, 56, 64, 256, 184, 192, 208, 176],
#  [56, 64, 72, 256, 192, 208, 208, 184],
#  [64, 56, 56, 72, 256, 176, 184, 184]
# ]



class wrangling:
    #Step:1 Color Space Conversion and Downsampling
    def ColorSpaceConversion(self, img):
        # Luminance
        Y   = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        # Blue
        Cb  = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
        # Red
        Cr  = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] -0.0813 * img[:,:,2] + 128
        return Y,Cb,Cr

    #Input Argument must be a 2D np array
    def Downsampling(self, img):
        x,y = img.shape[:2]
        downsampled = np.zeros((int(x/2),int(y/2)))

        for i in range (0,x-x %2,2):
            for j in range (0,y-y%2,2):
                downsampled[i//2,j//2] = (img[i+0,j+0] + img[i+0,j+1]+ img[i+1,j+0]+ img[i+1,j+1]) / 4 
        return downsampled
    
    
    
    def zigzag_traversal(self, matrix):
        n = len(matrix)
        result = []

        for d in range(2 * n - 1):
            temp = []

            # starting row for diagonal d
            r = 0 if d < n else d - n + 1
            # starting col for diagonal d
            c = d if d < n else n - 1

            while r < n and c >= 0:
                temp.append(matrix[r][c])
                r += 1
                c -= 1

            # reverse even diagonals (0,2,4,...)
            if d % 2 == 0:
                temp.reverse()

            result.extend(temp)

        return result

    
    def pack0ToList(self, res):
        """
        Pack consecutive zeros into [0, count] markers.
        Encoding scheme:
        - For non-zero values, append the value.
        - For a run of k zeros (k >= 1), append [0, k].
        This is applied to the whole flattened list.
        """
        ans = []
        i = 0
        n = len(res)
        while i < n:
            if res[i] == 0:
                # count consecutive zeros starting at i
                j = i
                while j < n and res[j] == 0:
                    j += 1
                count = j - i
                ans.extend([0, count])
                i = j
            else:
                ans.append(res[i])
                i += 1
        return ans

  
    def blockDCT(self, block):
        block = block.astype(float) - 128
        return dct(dct(block.T, norm='ortho').T, norm='ortho')


    def PaddingAndDCT(self,padded):
        DCT_padded = np.zeros_like(padded)
        for i in range(0, padded.shape[0], 8):
            for j in range(0, padded.shape[1], 8):
                DCT_padded[i:i+8,j:j+8] = Op.blockDCT(padded[i:i+8,j:j+8]) // Quantization_table
        return DCT_padded

    def pad(self, to_pad,x,y):
        pad_x = (8 - x % 8) % 8
        pad_y = (8 - y % 8) % 8
        return np.pad(to_pad, ((0,pad_x),(0,pad_y)), mode='constant', constant_values=0),x+pad_x,y+pad_y

    def blockify(self,mat):
        traversed8_8 = np.zeros(64,dtype=float)
        ans = []
        tem = []
        for i in range(0, mat.shape[0], 8):
            for j in range(0, mat.shape[1], 8):
                # print(mat[i:i+8,j:j+8])
                traversed8_8 = self.zigzag_traversal(mat[i:i+8,j:j+8])
                tem.extend(traversed8_8)
                ans.extend(self.pack0ToList(traversed8_8))
        return ans,tem



#Image as input
image = Image.open("trial.png")
image_rgb = image.convert("RGB")
image_array = np.array(image_rgb)


Op = wrangling()
Y,Cb,Cr = Op.ColorSpaceConversion(image_array)
Downsampled_Cb = Op.Downsampling(Cb)
Downsampled_Cr = Op.Downsampling(Cr)


target = Y
#Step:2 DCT and quantanization 
OriginalRow,OriginalCol = target.shape[0:2] #Preserving the dimension of the target before padding
#Using scippy fft to perform DCT
# DCT Padded    - DCT on the paded target where target  can by Y Cb or Cr. The padding is done so that we can have 8*8 blocks of whole image
# DCT          - Actual Discrete Cosine Transform of the whole picture
# imgdata      - Compressed Image Data

target_padded,PaddedRow,PaddedCol = Op.pad(target,OriginalRow,OriginalCol)
# print(OriginalRow,OriginalCol,"Padded to ---->",target_padded.shape)
DCT_padded = Op.PaddingAndDCT(target_padded)
DCT = DCT_padded[:OriginalRow,:OriginalCol]
print("size",DCT_padded.size)
#image = Image.fromarray(DCT)
imgdata,upack0 = Op.blockify(DCT_padded)
imgdata = np.array(imgdata,dtype=float)
upack0  = np.array(upack0,dtype=float)
print(f"{DCT_padded.size} --> {len(imgdata)}")
print("Compression ratio ",  len(imgdata)/DCT_padded.size)

img = Image.fromarray(target)
img.show()


# # k = DCT_padded[0, 0:100].astype(float)
# # print(k.tolist())
# # print("x"*100)
# # k = ans[0:100]
# # print([float(x) for x in k])



# # DCT = DCT_padded[:x,:y]
# # print(DCT)


# # image = Image.fromarray(DCT)
# # image.show()

# # import matplotlib.pyplot as plt
# # import numpy as np

# # # DCT_padded is already 2D
# # # Create a mask for zeros: 1 where value is zero, 0 otherwise
# # zeros_mask = (DCT_padded == 0).astype(int)

# # # Plot the heatmap
# # plt.figure(figsize=(12, 8))
# # plt.imshow(zeros_mask, cmap='gray', interpolation='nearest')
# # plt.colorbar(label='Zero presence (1=zero, 0=non-zero)')
# # plt.title("Heatmap of Zeros in DCT_padded")
# # plt.show()

class Decompression:
    def unPack0FromList(self, lst):
        """
        Decode the pack0ToList scheme above.
        """
        arr = []
        i = 0
        n = len(lst)
        while i < n:
            if lst[i] == 0:
                # must have a count after it
                if i + 1 >= n:
                    raise ValueError("Malformed input to unPack0FromList: trailing 0 with no count")
                count = int(lst[i+1])
                if count > 0:
                    arr.extend([0] * count)
                i += 2
            else:
                arr.append(lst[i])
                i += 1
        return arr


    def inverse_zigzag(self, arr, n):
        matrix = [[0]*n for _ in range(n)]
        index = 0
        for d in range(2 * n - 1):
            temp_positions = []
            r = 0 if d < n else d - n + 1
            c = d if d < n else n - 1
            while r < n and c >= 0:
                temp_positions.append((r, c))
                r += 1
                c -= 1
            if d % 2 == 0:
                temp_positions.reverse()
            for (r, c) in temp_positions:
                matrix[r][c] = arr[index]
                index += 1
        return matrix


    def reconstruct_imageblocks(self, unpacked_img_data, paddedRow, paddedCol):
        rows_of_blocks = paddedRow // 8
        cols_of_blocks = paddedCol // 8

        # Initialize big matrix
        bigMatrix = [[0 for _ in range(paddedCol)] for _ in range(paddedRow)]

        start = 0
        block_index = 0
        total_blocks = rows_of_blocks * cols_of_blocks

        for br in range(rows_of_blocks):
            for bc in range(cols_of_blocks):
                if start >= len(unpacked_img_data):
                    
                    break
                block = self.inverse_zigzag(unpacked_img_data[start:start+64], 8)
              
                block = [[float(x) for x in row] for row in block]  # convert to float
                start += 64

                # Place 8x8 block in bigMatrix
                for i in range(8):
                    for j in range(8):
                        bigMatrix[br*8 + i][bc*8 + j] = block[i][j]

                block_index += 1
                if block_index >= total_blocks:
                    break

        return bigMatrix
    
    def blockIDCT(self, block):
    # Apply 2D inverse DCT
        return idct(idct(block.T, norm='ortho').T, norm='ortho') + 128
    
    def IDCT(self,target):
        IDCT_target = np.zeros_like(target)
        for i in range(0, target.shape[0], 8):
            for j in range(0, target.shape[1], 8):
                IDCT_target[i:i+8,j:j+8] = self.blockIDCT(target[i:i+8,j:j+8]) * Quantization_table
        return IDCT_target
    




dc = Decompression()
unpackedImgData = dc.unPack0FromList(imgdata)  # your input list
unpackedImgData = np.array(unpackedImgData,dtype=float)

ImageBlocks = dc.reconstruct_imageblocks(unpackedImgData, PaddedRow, PaddedCol)
ImageBlocks = np.array(ImageBlocks, dtype=float)

# print(DCT_padded[16:32,16:32])
# print("-"*1000)
# print(ImageBlocks[16:32,16:32])
DecompressedIDCT = dc.IDCT(ImageBlocks)
img = Image.fromarray(DecompressedIDCT)
img.show()

