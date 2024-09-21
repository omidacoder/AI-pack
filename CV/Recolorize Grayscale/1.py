import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# reading image as grayscale
# img = cv.imread('fruit-basket-2.jpg',cv.IMREAD_GRAYSCALE)
img = cv.imread('fruit basket.jpg',cv.IMREAD_GRAYSCALE)
# reading image colorfull for help
# colorfullImg = cv.imread('fruit-basket-2.jpg')
colorfullImg = cv.imread('fruit basket.jpg')
colorfullImg = cv.cvtColor(colorfullImg , cv.COLOR_RGB2BGR)
blue_channel , green_channel , red_channel = cv.split(colorfullImg)
rows, cols = img.shape
R = np.zeros((rows , cols) , np.int64)
G = np.zeros((rows , cols) , np.int64)
B = np.zeros((rows , cols) , np.int64)
# enumarating pic
for (x, y), pixel in np.ndenumerate(img):
    # print(green_channel[x,y] * pixel / (red_channel[x,y] + 1))
    R[x,y] = pixel
    if(green_channel[x,y] - red_channel[x,y] < 256 - pixel):
        G[x,y] = green_channel[x,y] - red_channel[x,y] + pixel
    else:
        G[x,y] = 255
    if(blue_channel[x,y] - red_channel[x,y] < 256 - pixel):
        B[x,y] = blue_channel[x,y] - red_channel[x,y] + pixel
    else:
        B[x,y] = 255
    if(R[x,y] != 255 or G[x,y] != 255 or B[x,y] != 255):
        R[x,y] -= 20
        G[x,y] -= 20
        B[x,y] -= 20
result = cv.merge([B , G , R])
plt.subplot(221),plt.imshow(colorfullImg)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img, cmap = 'gray')
plt.title('GrayScale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(result, cmap = 'gray')
plt.title('Recolored Image'), plt.xticks([]), plt.yticks([])
plt.show()
