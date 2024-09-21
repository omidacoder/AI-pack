# importing needed libraries
import scipy.io
from scipy.linalg import svd
from matplotlib import pyplot as plt
import numpy as np
# reading mat file
mat = scipy.io.loadmat('mnistAll.mat')
# reading first 100 samples from each class of mat file
A = []
for c in mat.items():
    if "test" in c[0] or "__" in c[0]:
        continue
    for i in range(100):
        A.append(c[1][i])
# A is 1000 * 728 dimension
data_count = len(A)
dimension_count = 28*28
# centerign data
# calculating mean
mu = [ 0 for i in range(dimension_count)]
for i in range(data_count):
    for j in range(dimension_count):
        mu[j] += A[i][j]
for i in range(len(mu)):
    mu[i] /= data_count # number of data
# calculating APrime
for i in range(data_count):
    for j in range(dimension_count):
        A[i][j] -= mu[j]
# calculating SVD
U , S , VT = svd(A)
V = np.transpose(VT)
print("Transformation Matrix Is :")
print(VT)
# the transformation matrix is now VT
# plotting all singular values
Y = [0 for i in S]
plt.title("Singular Values")
plt.xlabel("singular values")
plt.xlim([0,5000])
plt.ylim([-1,1])
plt.plot(S,Y,'o',color='blue' , label='singular values')
plt.legend()
plt.show()
# calculating AZegond
APrime = np.array(A)
V = np.array(V)
Azegond = APrime.dot(V)
# now we first two columns of Azegond In PCA
# plotting these points
plotting = [[],[]]
for i in range(2):
    for j in range(0,1000,100):
        plotting[i].append(Azegond[j][i])
        plotting[i].append(Azegond[j+1][i])
fig , a = plt.subplots(2,5)
#lets plot all 10 subplots
for i in range(2):
    for j in range(5):
        a[i][j].plot([plotting[0][2*(i*5+j)],plotting[0][2*(i*5+j)+1]],[plotting[1][2*(i*5+j)] , plotting[1][2*(i*5+j)+1]],'o',color='red')
        a[i][j].set_title('Class '+str(i*5+j))
figManager = plt.get_current_fig_manager()
# change below line based on operating system and matplotlib backend
figManager.window.state("zoomed")
plt.show()
