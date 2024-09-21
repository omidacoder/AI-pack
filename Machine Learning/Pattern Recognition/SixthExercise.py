import math
from random import random
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from numpy.linalg import eig
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity as similarity

# این کلاس برای نگهداری آیگن وکتور ها استفاده میشود و قابلیت مقایسه دارد تا بتوانیم مقادیر آن را سورت کنیم و دومین وکتور آن را برای کلاسیفای استفاده کنیم
class eigenPair:
    def __init__(self , landa , vector , key) -> None:
        self.landa = landa
        self.vector = vector
        self.key = key
        pass
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o , eigenPair) and self.landa == __o.landa
    def __lt__(self , __o) -> bool:
        return isinstance(__o , eigenPair) and self.landa < __o.landa
    def __gt__(self , __o) -> bool:
        return isinstance(__o , eigenPair) and self.landa > __o.landa
# functions defenition
# از این تابع برای مپ کردن ساب پلات ها استفاده میشود
def getMap(i):
    if i == 5:
        return 1,2
    elif i == 4:
        return 1,1
    elif i == 3:
        return 1,0
    elif i == 2:
        return 0,2
    elif i == 1:
        return 0,1
    elif i == 0:
        return 0,0
    else:
        raise Exception("cannot map i")
# تابع شاخص رندوم ایندکس
def RI(predicted , reals):
    if len(predicted) != len(reals):
        raise Exception("predicted length is not equal to real labels length for calculating RI")
    a = 0 # pairs in same cluster in both predicted and reals
    b = 0 # pairs in different clusters in both predicted and reals
    c = 0 # pairs together in reals but different cluster in predicted 
    d = 0 # pairs together in predicted but different in reals
    for i in range(len(predicted)):
        for j in range(i+1,len(predicted)):
            # if i == j:
            #     continue
            predicted_same = False
            reals_same = False
            if predicted[i] == predicted[j]:
                predicted_same = True
            if reals[i] == reals[j]:
                reals_same = True
            if reals_same and predicted_same:
                a+=1
            if not reals_same and not predicted_same:
                b+=1
            if reals_same and not predicted_same:
                c+=1
            if not reals_same and predicted_same:
                d+=1
    return (a+b)/(a+b+c+d) # فرمول محاسبه رندوم ایندکس
def distance(first,second): # norm 2
    return (first[0]-second[0])**2 + (first[1] - second[1])**2
# این تابع برای ساخت گراف به روش knn استفاده میشود
def buildGraphKnn(dataPoints , k , sigma): # returns adjacency matrix and Degree Matrix
    A = np.zeros((len(dataPoints) , len(dataPoints))) # adjacency matrix
    D = np.zeros((len(dataPoints) , len(dataPoints))) # degree matrix
    for i in range(len(dataPoints)):
        A[i][i] = 0
        kn=0
        maximum_distance = 0
        avg_distance = 0
        # بدست آوردن همسایه های نزدیگ به این سمپل به تعداد k
        for j in range(len(dataPoints)):
            if j == i:
                continue
            d = distance(dataPoints[i] , dataPoints[j])
            if d > maximum_distance:
                maximum_distance = d
                avg_distance += d
                kn+=1
            if(kn == k):
                break
        avg_distance /= kn
        for j in range(i+1,len(dataPoints)):
            # اگر جزو همسایه های نزدیک باش وزن آن بر اساس فرمول موجود در اسلاید ها به دست می آید در غیر این صورت یالی قرار نمیگیرد
            if distance(dataPoints[i],dataPoints[j]) <= maximum_distance:
                A[i][j] = math.exp(-distance(dataPoints[i] , dataPoints[j]) / (sigma * avg_distance)**2)
                A[j][i] = math.exp(-distance(dataPoints[j] , dataPoints[i]) / (sigma * avg_distance)**2)
            else:
                A[i][j] = 0
                A[j][i] = 0
    # ساخت ماتریس درجه
    for i in range(len(dataPoints)):
        for j in range(len(dataPoints)):
            if i == j:
                continue
            D[i][i] += A[i][j]
    return A , D

# الگوریتم NCut
def NCutAlgorithm(dataPoints , k , sigma):
    # مانند اسلاید ها ابتدا آیگن وکتور ها و آیگن ولیو های ماتریس لاپلاسین را بدست می آوریم
    A , D = buildGraphKnn(dataPoints , k , sigma)
    L = D - A
    landa , v = eig(L)
    # making eigen values
    pairs = []
    for i in range(len(landa)):
        eigen = eigenPair(landa[i] , [] , i)
        for j in range(len(v)):
            eigen.vector.append(v[j][i])
        pairs.append(eigen)
    # now we can sort eigen values
    pairs.sort()
    # دومین آیگن وکتور را برمیداریم و بر اساس آن طبقه بندی را انجام میدهیم
    predicted = np.zeros(len(dataPoints))
    for i in range(len(pairs[1].vector)):
        if pairs[1].vector[i] <= 0:
            predicted[i] = 1
        else:
            predicted[i] = 0
    return predicted


n_samples = 400
samples, labels = make_circles(n_samples=n_samples, factor=.3, noise=.05)
x1 = []
y1 = []
x0 = []
y0 = []
for i in range(len(samples)):
    if labels[i] == 1:
        x1.append(samples[i][0])
        y1.append(samples[i][1])
    else:
        x0.append(samples[i][0])
        y0.append(samples[i][1])
# showing data points
plt.title("Values")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x1,y1,'+',color='red' , label='data samples class 1')
plt.plot(x0,y0,'o',color='blue' , label='data samples class 0')
plt.legend()
plt.show()
k = [3,5,7,10,15,20]
sigma = [0.1,0.5,1,2,5,10]
fig , a = plt.subplots(2,3)
for i in range(6):
    predicted = NCutAlgorithm(samples,k[i] , sigma[i])
    print("RI value for k={0} and sigma={1} is {2}".format(k[i] , sigma[i],RI(predicted , labels)))
    # ploting
    x1 = []
    y1 = []
    x0 = []
    y0 = []
    for j in range(len(samples)):
        if predicted[j] == 1:
            x1.append(samples[j][0])
            y1.append(samples[j][1])
        else:
            x0.append(samples[j][0])
            y0.append(samples[j][1])
    m , n = getMap(i)
    a[m][n].plot(x1,y1,'+',color='red')
    a[m][n].plot(x0,y0,'o',color='blue')
    a[m][n].set_title('k = {0}'.format(k[i]))
figManager = plt.get_current_fig_manager()
# change below line based on operating system and matplotlib backend
figManager.window.state("zoomed")
plt.show()

# ************ section 2 k-means ****************
k = 3
sigma = 1
# از این تابع برای یافتن میانگین کلاس ها در کی مینز استفاده میشود
def find_centroids(dataPoints , predicted):
    x1Avg = 0
    y1Avg = 0
    x0Avg = 0
    y0Avg = 0
    c1Count = 0
    c0Count = 0
    for i in range(len(dataPoints)):
        if predicted[i] == 1:
            c1Count +=1
            x1Avg += dataPoints[i][0]
            y1Avg += dataPoints[i][1]
        else:
            c0Count +=1
            x0Avg += dataPoints[i][0]
            y0Avg += dataPoints[i][1]
    x0Avg /= c0Count
    y0Avg /= c0Count
    x1Avg /= c1Count
    y1Avg /= c1Count
    return [x0Avg , y0Avg] , [x1Avg , y1Avg]
# از این تابع برای مقدار دهی دوباره به کلاس ها بر اساس مرکز های جدید در کی مینز استفاده میشود
def reassign(dataPoints , predicted , c0 , c1):
    for i in range(len(dataPoints)):
        if distance(dataPoints[i] , c0) > distance(dataPoints[i] , c1):
            predicted[i] = 1
        else:
            predicted[i] = 0

def k_means(dataPoints , k , sigma):
    buildGraphKnn(dataPoints , k , sigma)
    # random centroids
    c0 = [random() , random()]
    c1 = [random() , random()]
    predicted = np.zeros(len(dataPoints))
    reassign(dataPoints , predicted , c0 , c1)
    # assign dataSamples to centroids
    counter = 0
    # الگوریتم کی مینز پس از 150 مرحله مشخص تمام میشود که یکی از راه های پایان دادن به این الگوریتم است
    while(counter < 150): # static steps
        c0 , c1 = find_centroids(dataPoints , predicted)
        reassign(dataPoints , predicted , c0 , c1)
        counter+=1
    return predicted
predicted = k_means(samples , k , sigma)
print("RI value for k_means k={0} and sigma={1} is {2}".format(k , sigma,RI(predicted , labels)))
x1 = []
y1 = []
x0 = []
y0 = []
for j in range(len(samples)):
    if predicted[j] == 1:
        x1.append(samples[j][0])
        y1.append(samples[j][1])
    else:
        x0.append(samples[j][0])
        y0.append(samples[j][1])
plt.plot(x1,y1,'+',color='red')
plt.plot(x0,y0,'o',color='blue')
plt.suptitle('k_means')
plt.show()

# ************** section 3 Single Link And Complete Link ******************
# از این کلاس برای شکل دادن به داده ها استفاده میشود
class DataIndexPair:
    def __init__(self , data , index):
        self.data = data
        self.index = index
# تابع محاسبه بر اساس نزدیک ترین فاصله
def singleLinkDistance(c1 , c2):
    # shortest distance
    min_distance = 200000000
    for i in c1:
        for j in c2:
            if distance(i.data,j.data) < min_distance:
                min_distance = distance(i.data , j.data)
    return min_distance
# تابع محاسبه بر اساس دور ترین فاصله
def completeLinkDistance(c1 , c2):
    # shortest distance
    max_distance = -200000000
    for i in c1:
        for j in c2:
            if distance(i.data,j.data) > max_distance:
                max_distance = distance(i.data , j.data)
    return max_distance
# الگوریتم سلسله مراتبی که ابتدا هر داده را در یک کلاس جداگانه قرار میدهد و تعداد کلاس ها را کاهش میدهد تا به 2 کلاس برسد
def agglomerative(dataPoints : list , k , sigma , graphDistanceFunction):
    buildGraphKnn(dataPoints , k , sigma)
    c = []
    for i in range(len(dataPoints)):
        c.append([DataIndexPair(dataPoints[i] , i)])
    min_pair = (-1,-1)
    min_value = 200000000
    print(c[0][0].data[0])
    for i in range(len(c)):
        for j in range(i+1 , len(c)):
            dis = distance(c[i][0].data , c[j][0].data)
            if dis < min_value:
                min_pair = (i , j)
                min_value = dis
    for i in c[min_pair[1]]:
        c[min_pair[0]].append(i)
    del c[min_pair[1]]
    while len(c) > 2:
        print("number of clusters : " + str(len(c)))
        min_pair = (-1,-1)
        min_value = 200000000
        for i in range(len(c)):
            for j in range(i+1 , len(c)):
                # print(graphDistanceFunction(c[i] , c[j]))
                dis = graphDistanceFunction(c[i] , c[j])
                if  dis < min_value:
                    min_pair = (i , j)
                    min_value = dis
        # print(min_pair)
        for i in c[min_pair[1]]:
            c[min_pair[0]].append(i)
        del c[min_pair[1]]
    predicted = np.zeros(len(dataPoints))
    for i in c[0]:
        predicted[i.index] = 0
    for i in c[1]:
        predicted[i.index] = 1
    return predicted
    
predictedSingle = agglomerative(samples , k , sigma , singleLinkDistance)
predictedComplete = agglomerative(samples , k , sigma , completeLinkDistance)

print("RI value for singleLink k={0} and sigma={1} is {2}".format(k , sigma,RI(predictedSingle , labels)))
x1 = []
y1 = []
x0 = []
y0 = []
for j in range(len(samples)):
    if predictedSingle[j] == 1:
        x1.append(samples[j][0])
        y1.append(samples[j][1])
    else:
        x0.append(samples[j][0])
        y0.append(samples[j][1])
plt.plot(x1,y1,'+',color='red')
plt.plot(x0,y0,'o',color='blue')
plt.suptitle('singleLink agglomerative')
plt.show()

print("RI value for completeLink k={0} and sigma={1} is {2}".format(k , sigma,RI(predictedComplete , labels)))
x1 = []
y1 = []
x0 = []
y0 = []
for j in range(len(samples)):
    if predictedComplete[j] == 1:
        x1.append(samples[j][0])
        y1.append(samples[j][1])
    else:
        x0.append(samples[j][0])
        y0.append(samples[j][1])
plt.plot(x1,y1,'+',color='red')
plt.plot(x0,y0,'o',color='blue')
plt.suptitle('completeLink agglomerative')
plt.show()


        

    
    

