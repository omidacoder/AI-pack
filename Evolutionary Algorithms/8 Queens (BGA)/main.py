# Authors : Omid Davar , Mahdi Amiri Doomari
# Problem : Solve 8 Queens Using Binary Genetic Algorithm
import math
import random
import numpy as np
from matplotlib import pyplot as plt
# setting parameters
mating_pool_size = 10
mutation_probability = 0.001
crossover_probability = 0.9
stop_limit=2500
class binary_genetic_algorithm:
    def __init__(self,fittness_function , L) -> None:
        self.fittness_function = fittness_function
        self.L = L
        self.population = []
        self.mating_pool = []
        pass
    # check to see if doing something or not
    def dice_throw(self,P):
        if random.uniform(0,1) < P:
            return True
        else:
            return False
    def init_population(self):
        self.population = []
        for i in range(mating_pool_size):
            newRow = []
            for j in range(self.L):
                if random.uniform(0,1) > 0.5:
                    newRow.append(1)
                else:
                    newRow.append(0)
            self.population.append(newRow)
        pass
    # selection is done by `charkhe gardan`
    def selection(self):
        # sum of fitnesses to calculate probabilities
        sum = 0
        for i in range(mating_pool_size):
            sum += self.fittness_function(self.population[i])
        probabilities = []
        for i in range(mating_pool_size):
            probabilities.append(self.fittness_function(self.population[i]) / sum)
        q = random.uniform(0,1)
        sum = 0
        for i in range(mating_pool_size):
            if sum + probabilities[i] > q:
                return i
            sum += probabilities[i]
        return mating_pool_size - 1
    def fill_pool(self):
        self.mating_pool = []
        for i in range(mating_pool_size):
            self.mating_pool.append(self.population[self.selection()])
    def crossover(self):
        for i in range(0,mating_pool_size,2):
            if self.dice_throw(crossover_probability):
                # doing crossover operation
                parent1 = self.mating_pool[i]
                parent2 = self.mating_pool[i+1]
                cut_point = random.randint(0 , self.L - 1)
                child1 = parent1[:cut_point] + parent2[cut_point:]
                child2 = parent2[:cut_point] + parent1[cut_point:]
                self.mating_pool[i] = child1
                self.mating_pool[i+1] = child2;
        pass
    def mutation(self):
        for i in range(mating_pool_size):
            for j in range(self.L):
                if self.dice_throw(mutation_probability):
                    # toggling
                    if self.mating_pool[i][j] == 0:
                        self.mating_pool[i][j] = 1
                    else:
                        self.mating_pool[i][j] = 0
        pass
    def replace_population(self):
        self.population = self.mating_pool
        self.mating_pool = []
        pass
    def fitness_evaluation(self):
        sum = 0
        for i in range(mating_pool_size):
            sum += self.fittness_function(self.population[i])
        return sum / mating_pool_size
    def process(self):
        # doing steps until limitation
        self.init_population()
        self.iterations = []
        self.means = []
        self.best_so_far = []
        max = -math.inf
        for i in range(stop_limit):
            self.fill_pool()
            self.crossover()
            self.mutation()
            self.replace_population()
            # for plotting in future
            fit = self.fitness_evaluation()
            self.means.append(fit)
            self.iterations.append(i)
            if fit > max:
                max = fit
            self.best_so_far.append(max)
# use it in decoding chromosome
def decodeThreeBits(input):
    string_val = str(input[0]) + str(input[1]) + str(input[2])
    if string_val == "111":
        return 8;
    if string_val == "110":
        return 7;
    if string_val == "101":
        return 6;
    if string_val == "100":
        return 5;
    if string_val == "011":
        return 4;
    if string_val == "010":
        return 3;
    if string_val == "001":
        return 2;
    if string_val == "000":
        return 1;
def fitness(input):
    # result is queens thar threating each other that must be minimum so we use 1/f(x)
    # of Length 24 showing 8 queens locations : three bits per one location
    # extracting real locations from coded locations
    locations = []
    for i in range(0,24,3):
        locations.append(decodeThreeBits(input[i:i+3]))
    threats = 0;
    for i in range(8):
        # checking threats on ith queen
        for j in range(8):
            if i == j:
                continue
            if locations[i] == locations[j]:
                threats += 1
            if locations[i] == locations[j] + (j - i):
                threats += 1
            if locations[i] + (j - i) == locations[j]:
                threats += 1
    return 1/(threats*4+1) # + 1 is for division by zero escape 
def dev_fitness(input):
    # result is queens thar threating each other that must be minimum so we use 1/f(x)
    # of Length 24 showing 8 queens locations : three bits per one location
    # extracting real locations from coded locations
    locations = []
    
    for i in range(0,24,3):
        locations.append(decodeThreeBits(input[i:i+3]))
    print('locations :')
    print(locations)
    threats = 0;
    for i in range(8):
        # checking threats on ith queen
        for j in range(i,8):
            if i == j:
                continue
            if locations[i] == locations[j]:
                threats += 1
            if locations[i] + (j - i) == locations[j]:
                threats += 1
            if locations[i] == locations[j] + (j - i):
                threats += 1
    print("threats : " + str(threats))
# running all above
instance = binary_genetic_algorithm(fitness,24)
instance.process()
print('8 queens locations result population in BGA (decoded version) is :')
for i in range(len(instance.population)):
    dev_fitness(instance.population[i])
# plotting history
plt.title("Best So Far")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.plot(instance.iterations,instance.best_so_far,'b-' , label='best so far')
plt.show()
plt.title("Means")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.plot(instance.iterations,instance.means,'b-'  , label='means') 
plt.show()