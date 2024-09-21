# importing needed libraries
import math
import random
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
datasetPath='Data_Normal.txt'
glassDatasetPath='Glass.txt'
learningRate = 0.001 # can be changed
peopleEpochs = 150 # can be changed
glassEpochs = 500 # can be changed
# class declaration
class DatasetRow:
    def __init__(self , attributeValues , label):
        self.attributeValues = attributeValues
        self.label = label
class Neuron: 
    # all neurons has sigmoid activation function 
    # so class with bigger sigmoid value is evaluated class by the network
    def __init__(self , numberOfInputs : int):
        self.weights = [0 for i in range(numberOfInputs + 1)]
    def sigmoid(self,input):
        return 1 / (1 + math.e**(-input))
    def evaluate(self , inputs : list):
        if len(inputs) != len(self.weights) - 1:
            raise Exception()
        inputOfNeuron = 0
        for i in range(len(inputs)):
            inputOfNeuron += inputs[i] * self.weights[i + 1]
        # and the bias
        inputOfNeuron += self.weights[0]
        return self.sigmoid(inputOfNeuron)
class Model: # the model has 10 neurons for glass and 2 neurons for people dataset
    # model has only one layer
    def __init__(self,numberOfInputs : int,numberOfClasses : int):
        self.neurons = [Neuron(numberOfInputs) for i in range(numberOfClasses)]
        self.numberOfInputs = numberOfInputs;
        self.numberOfClasses = numberOfClasses
    def train(self , trainData : list , epochs):
        # updates are incremental
        for e in range(epochs):
            for d in trainData:
                for n in range(len(self.neurons)):
                    evaluated = self.neurons[n].evaluate(d.attributeValues)
                    y = 0
                    if d.label == n:
                        y = 1
                    # update weights
                    for w in range(len(self.neurons[n].weights)):
                        if w == 0:
                            self.neurons[n].weights[w] += (learningRate) * (y - evaluated)
                        self.neurons[n].weights[w] += (learningRate) * (y - evaluated) * d.attributeValues[w - 1]
    def predict(self , row : list):
        max_neuron_index = -1
        max_evaluated = float('-inf')
        for n in range(len(self.neurons)):
            eval = self.neurons[n].evaluate(row)
            if eval > max_evaluated:
                max_evaluated = eval
                max_neuron_index = n
        return max_neuron_index
    def evaluate(self , testData : list):
        y_true = []
        y_pred = []
        for d in testData:
            y_pred.append(self.predict(d.attributeValues))
            y_true.append(d.label)
        return y_true , y_pred
        
# declaring datasets
dataset = []
glassDataset = []
# declaring functions
def splitDataset(dataset):
    train_set = dataset[:math.floor(len(dataset) * 70 / 100)]
    test_set = dataset[math.floor(len(dataset) * 70 / 100) + 1:]
    return train_set , test_set
# reading Dataset_Normal.txt
file = open(datasetPath , 'r')
# skipping first line
file.readline()
nextLine = file.readline()
while(nextLine != ""):
    splitted = nextLine.split(";")
    # converting read line to object of class
    row = DatasetRow([float(splitted[0]),float(splitted[1])],int(splitted[2]))
    dataset.append(row)
    nextLine = file.readline()
# reading Glass.txt
file = open(glassDatasetPath , 'r')
nextLine = file.readline()
while(nextLine != ""):
    splitted = nextLine.split('	')
    # converting read line to object of class
    row = []
    for i in range(len(splitted)-1):
        row.append(int(splitted[i]))
    datasetRow = DatasetRow(row , int(splitted[len(splitted) - 1]))
    glassDataset.append(datasetRow)
    nextLine = file.readline()
# split data to train and test
random.shuffle(dataset)
random.shuffle(glassDataset)
dataset_train , dataset_test = splitDataset(dataset)
glass_dataset_train , glass_dataset_test = splitDataset(glassDataset)
# lets make and train model
dataset_model = Model(2 , 2)
glass_dataset_model = Model(11 , 7)
dataset_model.train(dataset_train , peopleEpochs)
glass_dataset_model.train(glass_dataset_train , glassEpochs)
y_true , y_pred = dataset_model.evaluate(dataset_test)
#report dataset metrics
print("results for people dataset:")
print('accuracy : ',accuracy_score(y_true , y_pred) * 100 , '%')
print('f1-score : ' , f1_score(y_true , y_pred , average='weighted' , zero_division=0) * 100 , '%' )
print('precision : ' , precision_score(y_true , y_pred , average='weighted' ,zero_division=0) * 100 , '%')
print('recall : ' , recall_score(y_true , y_pred , average='weighted' , zero_division=0) * 100 , '%')
y_true , y_pred = glass_dataset_model.evaluate(glass_dataset_test)
#report glass dataset metrics
print('results for glass dataset:')
print('accuracy : ',accuracy_score(y_true , y_pred) * 100 , '%')
print('f1-score : ' , f1_score(y_true , y_pred , average='weighted' , zero_division=0) * 100 , '%' )
print('precision : ' , precision_score(y_true , y_pred , average='weighted' ,zero_division=0) * 100 , '%')
print('recall : ' , recall_score(y_true , y_pred , average='weighted' ,zero_division=0) * 100 , '%')


