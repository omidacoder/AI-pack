import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#reading image
img = cv.imread('woods.jpg',cv.IMREAD_GRAYSCALE)
transform = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# centering 
centered_transform = np.fft.fftshift(transform)
phase = np.angle(centered_transform[:,:,0])
# getting the real part
magnitude_spectrum = np.abs(centered_transform[:,:,0])
magnitude_spectrum_log = np.log(magnitude_spectrum + 1)
# getting real image from frequency domain
decentered = np.fft.ifftshift(centered_transform)
result = cv.idft(decentered)
result = cv.magnitude(result[:, :, 0], result[:, :, 1])
result_centered = cv.idft(centered_transform)
result_centered = cv.magnitude(result_centered[:,:,0] , result_centered[:,:,1])
plt.subplot(221),plt.imshow(img , cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(phase, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(221),plt.imshow(img , cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(magnitude_spectrum_log, cmap = 'gray')
plt.title('Logarithm Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(221),plt.imshow(img , cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(result_centered, cmap = 'gray')
plt.title('Centered Inverse'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(result, cmap = 'gray')
plt.title('Decentered Inverse'), plt.xticks([]), plt.yticks([])
plt.show()