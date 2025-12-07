# JPEG Image Compression in Python

## Overview

This project demonstrates a Python implementation of **JPEG image compression**. It includes color space conversion, chroma subsampling, block-based Discrete Cosine Transform (DCT), and quantization. The project uses **NumPy**, **Pillow**, and **SciPy**, and serves as an educational example of image compression.

---

## Project Structure

* `Compression.py` – Compresses an input image and stores the result in `compressed_imgdata.npz`.
* `Decompression.py` – Loads `compressed_imgdata.npz` and reconstructs the image.
* `demo.png` – Example input image.
* `requirements.txt` – Python dependencies.

---

## Environment Setup

1. **Create a virtual environment**:

```bash
python -m venv venv
```

2. **Activate the environment**:

*Windows:*

```bash
venv\Scripts\activate
```

*Linux/Mac:*

```bash
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage

### Compression

1. Open `Compression.py` and set the correct path to your input image.
2. Run the script:

```bash
python Compression.py
```

3. The compressed data will be saved as `compressed_imgdata.npz`.

### Decompression

1. Ensure `compressed_imgdata.npz` is present in the working directory.
2. Run the script:

```bash
python Decompression.py
```

3. The original image will be reconstructed from the compressed data.

---

## Implementation Details

1. **Color Space Conversion**

```python
Y   = 0.299 * R + 0.587 * G + 0.114 * B
Cb  = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr  = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
```

2. **Chroma Subsampling**
   4:2:0 subsampling is applied by reducing the resolution of Cb and Cr channels by half in both dimensions.

3. **Block-based DCT**
   Each 8×8 block is transformed using 2D DCT:

```python
blockDCT = dct(dct(block.T, norm='ortho').T, norm='ortho')
```

4. **Quantization**
   DCT coefficients are quantized using a predefined table to reduce data size and remove perceptually less important information.

---

## Notes

* **Compression.py** handles image compression and stores the result in `compressed_imgdata.npz`.
* **Decompression.py** retrieves the compressed data and reconstructs the image.
* The project currently supports grayscale and YCbCr channels; future work can include **Huffman encoding** and full **color reconstruction**.

---
