# JPEG Image Compression in Python

## Overview

This repository contains a Python implementation of the **JPEG image compression algorithm**. The project demonstrates the core steps of JPEG compression, including color space conversion, chroma subsampling, block-based Discrete Cosine Transform (DCT), and quantization. It serves as an educational example for understanding image compression at a low level.

The implementation uses **NumPy**, **Pillow**, and **SciPy** to handle image processing and DCT computation.

---

## Features

- **Color Space Conversion:** Converts RGB images into YCbCr color space to separate luminance and chrominance components.
- **Chroma Subsampling:** Reduces the resolution of the Cb and Cr channels to exploit human visual perception.
- **Block-based DCT:** Divides the image into 8×8 blocks and applies 2D DCT using SciPy's FFT-based implementation.
- **Quantization:** Applies a custom quantization table to compress the frequency coefficients and reduce data size.
- **Padding:** Handles images with dimensions not divisible by 8 by padding them appropriately.

---

## Project Structure

- `main.py` – Python script implementing JPEG compression up to DCT and quantization.
- `demo.png` – Example input image used for testing (user-supplied).
- `README.md` – Project documentation.
- `requirements.txt` – Python dependencies for environment setup.

---

## Environment Setup

1. **Create a virtual environment** (recommended):

```bash
python -m venv venv
````

2. **Activate the environment**:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux / Mac:**

```bash
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```


---

## Usage

1. Clone the repository:

```bash
git clone <repository_url>
cd jpeg-compression-python
```

2. Ensure the virtual environment is activated and dependencies are installed.

3. Run the compression script:

```bash
python main.py
```

4. The script will display the downsampled chrominance channel and the quantized DCT coefficients of the luminance channel.

---

## Implementation Details

1. **Color Space Conversion**

```python
Y   = 0.299 * R + 0.587 * G + 0.114 * B
Cb  = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
Cr  = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
```

2. **Chroma Subsampling**

* 4:2:0 subsampling is applied by averaging each 2×2 block in the Cb and Cr channels.

3. **DCT**

* Each 8×8 block of the luminance channel is shifted by 128 and transformed using 2D DCT:

```python
blockDCT = dct(dct(block.T, norm='ortho').T, norm='ortho')
```

4. **Quantization**

* The DCT coefficients are divided by a predefined quantization table to reduce data size and remove perceptually less important information.

---

## Future Work

* Implement **Huffman encoding** for entropy compression.
* Implement **inverse JPEG** to reconstruct the image from quantized DCT coefficients.
* Extend to full **color image reconstruction** with inverse chroma upsampling.

---





