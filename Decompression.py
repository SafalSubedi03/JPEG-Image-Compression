from PIL import Image
import numpy as np
import math
from scipy.fftpack import dct
from scipy.fftpack import idct
import matplotlib.pyplot as plt
from skimage.transform import resize


Quantization1 = [
 [1, 1, 1, 1, 2, 2, 3, 3],
 [1, 1, 1, 2, 2, 3, 3, 3],
 [1, 1, 2, 2, 3, 3, 3, 3],
 [1, 2, 2, 3, 3, 4, 4, 3],
 [2, 2, 3, 3, 3, 4, 4, 4],
 [2, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 3, 4, 4, 4, 4]
]

Quantization2 = \
[[1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1]]




Quantanization3 = [
 [ 8,  8,  8, 16, 24, 40, 56, 64],
 [ 8,  8, 16, 16, 32, 56, 64, 56],
 [ 8, 16, 16, 24, 40, 64, 72, 56],
 [16, 16, 24, 32, 56, 176, 248, 72],
 [24, 32, 40, 56, 72, 200, 192, 256],
 [40, 56, 64, 256, 184, 192, 208, 176],
 [56, 64, 72, 256, 192, 208, 208, 184],
 [64, 56, 56, 72, 256, 176, 184, 184]
]

Quantization_table = {1:Quantization1,2:Quantization2,3:Quantization2}

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
    
    def IDCT(self,target,element):
        IDCT_target = np.zeros_like(target)
        for i in range(0, target.shape[0], 8):
            for j in range(0, target.shape[1], 8):
                IDCT_target[i:i+8,j:j+8] = self.blockIDCT(target[i:i+8,j:j+8]) * Quantization2
        return IDCT_target
    


data = np.load("compressed_imgdata.npz")
dc = Decompression()
decompressed_components = {}  # To hold 2D arrays of each component

for i, name in enumerate(["Y", "Cb", "Cr"], start=1):
    # Retrieve stored data
    imgdata = np.array(data[f"{i}_imgdata"], dtype=float)
    PaddedRow = int(data[f"{i}_PaddedRow"])
    PaddedCol = int(data[f"{i}_PaddedCol"])

    # Decompression steps
    unpackedImgData = dc.unPack0FromList(imgdata)
    unpackedImgData = np.array(unpackedImgData, dtype=float)

    ImageBlocks = dc.reconstruct_imageblocks(unpackedImgData, PaddedRow, PaddedCol)
    ImageBlocks = np.array(ImageBlocks, dtype=float)

    DecompressedIDCT = dc.IDCT(ImageBlocks,i)

    # Store the decompressed 2D array
    decompressed_components[name] = DecompressedIDCT


Y = decompressed_components["Y"]
Cb = decompressed_components["Cb"]
Cr = decompressed_components["Cr"]

height,width = Y.shape

Cb_up = resize(Cb, (height, width), order=1, mode='reflect', anti_aliasing=False)
Cr_up = resize(Cr, (height, width), order=1, mode='reflect', anti_aliasing=False)

Y = Y.astype(float)
Cb_up = Cb_up.astype(float)
Cr_up = Cr_up.astype(float)


# Convert to RGB
R = Y + 1.402 * (Cr_up - 128)
G = Y - 0.344136 * (Cb_up - 128) - 0.714136 * (Cr_up - 128)
B = Y + 1.772 * (Cb_up - 128)

RGB = np.stack([R, G, B], axis=2)
RGB = np.clip(RGB, 0, 255).astype(np.uint8)
img = Image.fromarray(RGB)
img.show()