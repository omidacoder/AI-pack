import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#constants 
low_high_tresholds = [10,50,100,250]
bandpass_inner_treshold = 35
bandpass_outer_treshold = 70
#reading image
img = cv.imread('peacock-feather.jpg');
red_channel , green_channel , blue_channel = cv.split(img)
# functions
# returns inversed transform result
def distance(r1,c1,r2,c2):
    return math.sqrt((r1-r2)**2 + (c1-c2)**2)
def low_pass_filter(img,transform , treshold):
    # getting length of image matrix
    rows, cols = img.shape
    # getting vertical and horizontal centers
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if(distance(crow,ccol,i,j) < treshold):
                mask[i,j] = 1
    # product of image and mask
    f = transform * mask
    #inverse fourier transform
    decentered = np.fft.ifftshift(f)
    result = cv.idft(decentered)
    result = cv.magnitude(result[:, :, 0], result[:, :, 1])
    return result
# returns inversed transform result
def high_pass_filter(img,transform , treshold):
    # getting length of image matrix
    rows, cols = img.shape
    # getting vertical and horizontal centers
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if(distance(crow,ccol,i,j) < treshold):
                mask[i,j] = 0
    # product of image and mask
    f = transform * mask
    #inverse fourier transform
    decentered = np.fft.ifftshift(f)
    result = cv.idft(decentered)
    result = cv.magnitude(result[:, :, 0], result[:, :, 1])
    return result
# returns inversed transform result
def band_pass_filter(img,transform , high_treshold , low_treshold):
    # getting length of image matrix
    rows, cols = img.shape
    # getting vertical and horizontal centers
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if(distance(crow,ccol,i,j) < high_treshold and distance(crow,ccol,i,j) > low_treshold) :
                mask[i,j] = 0
    # product of image and mask
    f = transform * mask
    #inverse fourier transform
    decentered = np.fft.ifftshift(f)
    result = cv.idft(decentered)
    result = cv.magnitude(result[:, :, 0], result[:, :, 1])
    return result
# combine three channels
def combine(red,green,blue,filter_type,low_treshold , high_treshold):
    red_filtered = None
    green_filtered = None
    blue_filtered = None
    if filter_type == 'L':
        red_filtered=low_pass_filter(red_channel,red,low_treshold)
        green_filtered=low_pass_filter(green_channel,green,low_treshold)
        blue_filtered=low_pass_filter(blue_channel,blue,low_treshold)
    if filter_type == 'H':
        red_filtered=high_pass_filter(red_channel,red,high_treshold)
        green_filtered=high_pass_filter(green_channel,green,high_treshold)
        blue_filtered=high_pass_filter(blue_channel,blue,high_treshold)
    if filter_type == "B":
        red_filtered=band_pass_filter(red_channel,red,high_treshold,low_treshold)
        green_filtered=band_pass_filter(green_channel,green,high_treshold,low_treshold)
        blue_filtered=band_pass_filter(blue_channel,blue,high_treshold,low_treshold)
    # return red_filtered
    res = cv.merge([red_filtered , green_filtered , blue_filtered])
    res = res/np.amax(res)
    res = np.clip(res, 0, 1)
    plt.imshow(res)
    return (res*255).astype(np.uint8)
    



red_transform = cv.dft(np.float32(red_channel), flags=cv.DFT_COMPLEX_OUTPUT)
green_transform = cv.dft(np.float32(green_channel), flags=cv.DFT_COMPLEX_OUTPUT)
blue_transform = cv.dft(np.float32(blue_channel), flags=cv.DFT_COMPLEX_OUTPUT)
red_centered_transform = np.fft.fftshift(red_transform)
green_centered_transform = np.fft.fftshift(green_transform)
blue_centered_transform = np.fft.fftshift(blue_transform)
#low pass
low_result_0 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"L",low_high_tresholds[0],None)
low_result_1 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"L",low_high_tresholds[1],None)
low_result_2 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"L",low_high_tresholds[2],None)
low_result_3 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"L",low_high_tresholds[3],None)
#high pass
high_result_0 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"H",None,low_high_tresholds[0])
high_result_1 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"H",None,low_high_tresholds[1])
high_result_2 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"H",None,low_high_tresholds[2])
high_result_3 = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"H",None,low_high_tresholds[3])
#band pass
band_result = combine(red_centered_transform,green_centered_transform,blue_centered_transform,"B",bandpass_inner_treshold , bandpass_outer_treshold)
plt.axis('off')
plt.subplot(231),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(low_result_0)
plt.title('Low Pass 10'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(low_result_1)
plt.title('Low Pass 50'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(low_result_2)
plt.title('Low Pass 100'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(low_result_3)
plt.title('Low Pass 250'), plt.xticks([]), plt.yticks([])
plt.show()

plt.axis('off')
plt.subplot(231),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(high_result_0)
plt.title('High Pass 10'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(high_result_1)
plt.title('High Pass 50'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(high_result_2)
plt.title('High Pass 100'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(high_result_3)
plt.title('High Pass 250'), plt.xticks([]), plt.yticks([])
plt.show()

plt.axis('off')
plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(band_result)
plt.title('Band Pass'), plt.xticks([]), plt.yticks([])
plt.show()

