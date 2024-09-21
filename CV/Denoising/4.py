import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#reading image
img = cv.imread('city.jpg',cv.IMREAD_GRAYSCALE)

transform = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
centered_transform = np.fft.fftshift(transform)
magnitude_spectrum = np.log(np.abs(centered_transform[:,:,1]))
# getting length of image matrix
rows, cols = img.shape
# getting vertical and horizontal centers
crow, ccol = int(rows / 2), int(cols / 2)
# making notch mask == band pass in this section becuase noise is in special destances from center in frequency domain
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 120:crow + 120, ccol - 120:ccol + 120] = 0
mask[crow - 90:crow + 90, ccol - 90:ccol + 90] = 1
# mask[crow + 120:,:] = 1
# mask[:,crow + 120:] = 1
# product of image and mask
f = centered_transform * mask
#inverse fourier transform
decentered = np.fft.ifftshift(f)
result = cv.idft(decentered)
result = cv.magnitude(result[:, :, 0], result[:, :, 1])
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(result, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(mask[:,:,0], cmap = 'gray')
plt.title('Notch Mask'), plt.xticks([]), plt.yticks([])
plt.show()

# second part 
#reading image
img = cv.imread('man.png',cv.IMREAD_GRAYSCALE)

transform = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
centered_transform = np.fft.fftshift(transform)
magnitude_spectrum = np.abs(centered_transform[:,:,0])
# getting length of image matrix
rows, cols = img.shape
# getting vertical and horizontal centers
crow, ccol = int(rows / 2), int(cols / 2)
# making notch mask
mask = np.ones((rows, cols, 2), np.uint8)
mask[:crow - 40, ccol - 20:ccol + 20] = 0
mask[crow + 40:, ccol - 20:ccol + 20] = 0
# product of image and mask
f = centered_transform * mask
#inverse fourier transform
decentered = np.fft.ifftshift(f)
result = cv.idft(decentered)
result = cv.magnitude(result[:, :, 0], result[:, :, 1])
plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(result, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(mask[:,:,0], cmap = 'gray')
plt.title('Notch Mask'), plt.xticks([]), plt.yticks([])
plt.show()