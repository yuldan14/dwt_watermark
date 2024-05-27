import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

current_path = str(os.path.dirname(__file__))  

image = 'yuldane.jpg'   
watermark = 'wm.jpg' 

def convert_image(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img = img.convert('L')
    img.save(image_name)

    image_array = np.array(img.getdata(), dtype=np.uint8).reshape((size, size))
    print(image_array[0][0])               
    print(image_array[10][10])             

    return image_array

def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    return coeffs

def embed_mod2(coeff_image, coeff_watermark, offset=0):
    for i in range(len(coeff_watermark)):
        for j in range(len(coeff_watermark[i])):
            coeff_image[i*2+offset][j*2+offset] = coeff_watermark[i][j]

    return coeff_image

def embed_mod4(coeff_image, coeff_watermark):
    for i in range(len(coeff_watermark)):
        for j in range(len(coeff_watermark[i])):
            coeff_image[i*4][j*4] = coeff_watermark[i][j]

    return coeff_image

def embed_watermark(watermark_array, orig_image):
    watermark_array_size = len(watermark_array)
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range(0, len(orig_image), 8):
        for y in range(0, len(orig_image), 8):
            if ind < len(watermark_flat):
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1 

    return orig_image

def apply_dct(image_array):
    size = image_array.shape[0]
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct

def inverse_dct(all_subdct):
    size = all_subdct.shape[0]
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct

def get_watermark(dct_watermarked_coeff, watermark_size):
    subwatermarks = []

    for x in range(0, dct_watermarked_coeff.shape[0], 8):
        for y in range(0, dct_watermarked_coeff.shape[0], 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark

def recover_watermark(image_array, model='haar', level=1):
    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    watermark_array = get_watermark(dct_watermarked_coeff, 128)
    watermark_array = np.uint8(watermark_array)

    # Save result
    img = Image.fromarray(watermark_array)
    img.save('recovered_watermark.jpg')

def print_image_from_array(image_array, name):
    image_array_copy = np.clip(image_array, 0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    
    # Jika citra grayscale, ubah ke mode RGB
    if len(image_array_copy.shape) == 2:
        img = img.convert('RGB')
    
    img.save(name)
def w2d(image_path):
    model = 'haar'
    level = 1
    image_array = convert_image(image_path, 2048)
    watermark_array = convert_image(watermark, 128)

    coeffs_image = process_coefficients(image_array, model, level=level)
    dct_array = apply_dct(coeffs_image[0])
    dct_array = embed_watermark(watermark_array, dct_array)
    coeffs_image[0] = inverse_dct(dct_array)
  
    # Reconstruction
    image_array_H = pywt.waverec2(coeffs_image, model)
    print_image_from_array(image_array_H, 'image_with_watermark.jpg')

    # Recover images
    recover_watermark(image_array=image_array_H, model=model, level=level)

w2d(image)
