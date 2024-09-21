# importing needed libraries and declare global variables
import numpy as np
from matplotlib import pyplot as plt
datasetPath='Data_Normal.txt'
datasetWithOutlierPath='Data_With_Outlier.txt'
learningRate = 0.009 # can be changed
epochs = 2000 #can be changed
# classes declaration
class DatasetRow:
    def __init__(self , height : float,weight : float , Sex : float):
        self.height = height
        self.weight = weight
        self.Sex = Sex
class Dataset:
    def __init__(self):
        self.data = []
        self.t = [1,1,1]
    def append(self , row : DatasetRow):
        self.data.append(row)
    def normalize(self):
        # finding min and max of two attributes
        heightMax = float('-inf')
        heightMin = float('inf')
        weightMax = float('-inf')
        weightMin = float('inf')
        for row in self.data:
            if row.height > heightMax :
                heightMax = row.height
            if row.height < heightMin :
                heightMin = row.height
            if row.weight > weightMax :
                weightMax = row.weight
            if row.weight < weightMin :
                weightMin = row.weight
        # normalize data using minMax formula
        for row in self.data:
            row.height = (row.height - heightMin) / (heightMax - heightMin)
            row.weight = (row.weight - weightMin) / (weightMax - weightMin)
    def h(self , row : DatasetRow ):
        # hypothesis linear function
        return self.t[0] + self.t[1]*row.height + self.t[2]*row.weight
    def stochasticGradientDescent(self , row : DatasetRow):
        # t is tetha and the list has 3 elements
        # for stochastic we get just 1 DatasetRow and update t array
        self.t[0] += 2 * (learningRate / len(self.data)) * (row.Sex - self.h(row)) # this is like bias
        self.t[1] += 2 * (learningRate / len(self.data)) * (row.Sex - self.h(row)) * row.height
        self.t[2] += 2 * (learningRate / len(self.data)) * (row.Sex - self.h(row)) * row.weight
        # t values updated
    def batchGradientDescent(self):
        t0Sum = 0
        t1Sum = 0
        t2Sum = 0
        # calculating summation
        for d in self.data:
            t0Sum += (d.Sex - self.h(d))
            t1Sum += (d.Sex - self.h(d)) * d.height
            t2Sum += (d.Sex - self.h(d)) * d.weight
        # updating parameters after iteration of all data
        self.t[0] += (learningRate / len(self.data)) * t0Sum
        self.t[1] += (learningRate / len(self.data)) * t1Sum
        self.t[2] += (learningRate / len(self.data)) * t2Sum


    def linearRegression(self , type : str):
        # hypothesis for linear regression is : t0 + t1x1 + t2x2 = h(t)
        # cost function is : 1/m*Sum((h(t) - y)**2)
        # so the update rule is : t[new] = t[old] + (2 * alpha / m) * ( y - h(t[old])) * x
        # running stochastic gradient descent first
        # init t arrays
        if type == 'stochastic':
            for i in range(epochs):
                for row in self.data:
                    self.stochasticGradientDescent(row)
        else:
            for i in range(epochs):
                self.batchGradientDescent()

    def plot(self , title):
        # x is height
        # y is weight
        xMalePoints = []
        yMalePoints = []
        xFemalePoints = []
        yFemalePoints = []
        for row in self.data:
            if row.Sex > 0.5:
                xMalePoints.append(row.height)
                yMalePoints.append(row.weight)
            else:
                xFemalePoints.append(row.height)
                yFemalePoints.append(row.weight)

        Xmale = np.array(xMalePoints)
        Ymale = np.array(yMalePoints)
        Xfemale = np.array(xFemalePoints)
        YFemale = np.array(yFemalePoints)
        plt.title(title)
        plt.xlabel("height")
        plt.ylabel("weight")
        # plt.grid()
        plt.plot(Xmale,Ymale,'o',color='blue' , label='male')
        plt.plot(Xfemale,YFemale,'o' , color='red' , label='female')
        # plotting the line
        axes = plt.gca()
        # intercept = arz az mabda = -t0/t2
        # slope = shibe khat = t1/t2
        intercept = -self.t[0]/self.t[2]
        slope = self.t[1]/self.t[2]
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals,y_vals,'-',color='green',label='regression')
        plt.xlim([-0.25,1.25])
        plt.ylim([-0.25,1.25])
        plt.legend()
        plt.show()
    def clearWeights(self):
        self.t = [0,0,0]

# get initial instances of Dataset Class
dataset = Dataset()
datasetWithOutlier = Dataset()
# function declaration
def readFromTxtFile(path : str):
    file = open(path , 'r')
    # skipping first line
    file.readline()
    nextLine = file.readline()
    while(nextLine != ""):
        splitted = nextLine.split(";")
        # converting read line to object of class
        row = DatasetRow(float(splitted[0]),float(splitted[1]),float(splitted[2]))
        if path == "Data_Normal.txt":
            dataset.append(row)
        else:
            datasetWithOutlier.append(row)
        nextLine = file.readline()
readFromTxtFile(datasetPath)
# for stochastic mode
dataset.normalize()
dataset.linearRegression('stochastic')
dataset.plot("without outlier in stochastic mode")
# for batch mode
dataset.clearWeights()
dataset.linearRegression('batch')
dataset.plot("without outlier in batch mode")
# for the second dataset
readFromTxtFile(datasetWithOutlierPath)
datasetWithOutlier.normalize()
# stochastic mode
datasetWithOutlier.linearRegression('stochastic')
datasetWithOutlier.plot('with outlier in stochastic mode')
# batch mode
datasetWithOutlier.clearWeights()
datasetWithOutlier.linearRegression('batch')
datasetWithOutlier.plot('with outlier in batch mode')
