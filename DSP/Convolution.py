# Author : Omid Davar
# Subject : 1d convovle implementation
# Student Number : 400155017
import numpy as np
import matplotlib.pyplot as plt
# creating example input
y1 = np.array(range(1,10,2))
y2 = np.array(np.ones((5)))
print("y1 is :" , y1)
print("y2 is :" , y2)
x1 = range(len(y1))
x2 = range(len(y2))
plt.stem(x1 , y1 , 'o')
plt.title('x1 input 1d array')
plt.show()
plt.stem(x2 , y2 , 'o')
plt.title('x2 input 1d array')
plt.show()
# doing convolution
result = np.convolve(y1, y2)
print("result is : " , result)
resultx = range(len(result))
plt.stem(resultx , result , 'o')
plt.title('convolution result 1d array')
plt.show()




