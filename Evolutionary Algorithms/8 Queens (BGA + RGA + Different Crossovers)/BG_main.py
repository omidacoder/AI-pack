# Authors : Omid Davar , Mohammad Mahdi Amiri Doomari
# Problem : Solve 8 Queens Using Binary Genetic Algorithm
import math
import random
import numpy as np
from matplotlib import pyplot as plt

# setting parameters
mating_pool_size = 10
mutation_probability = 0.001
crossover_probability = 0.9
stop_limit = 2500

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

    #find mask for mask based cross over
    def mask(self):

        pattern_mask = []
        positive_mask = []
        negative_mask = []
        positive_population = []
        negative_population = []

        my_dict = dict()

        sum = 0

        for i in range(mating_pool_size):
            sum += self.fittness_function(self.population[i])
            my_dict[i] = self.fittness_function(self.population[i])

        #sort population by fitness
        ranked = sorted(my_dict.items(), key =
             lambda kv:(kv[1], kv[0]))

        sz = int(mating_pool_size/4)

        #select N/4 of worst of population
        for i in range(0, sz):
            negative_population.append(self.mating_pool[ranked[i][0]])

        #select N/4 of best of population
        for i in range(mating_pool_size - sz, mating_pool_size):
            positive_population.append(self.mating_pool[ranked[i][0]])


        for j in range(0, self.L):
            p_cnt_zero = 0
            p_cnt_one = 0

            n_cnt_zero = 0
            n_cnt_one = 0

            for i in range (0, sz):
                p = positive_population[i]
                n = negative_population[i]
                
                #counting number of 0 and 1 in i-th bit
                if p[j] == 1 :
                   p_cnt_one += 1 
                if p[j] == 0 :
                   p_cnt_zero += 1

                if n[j] == 1 :
                   n_cnt_one += 1 
                if n[j] == 0 :
                   n_cnt_zero += 1 
   
            if p_cnt_one >= p_cnt_zero:
                positive_mask.append(int(1))
            else:
                positive_mask.append(int(0))

            if n_cnt_one >= n_cnt_zero:
                negative_mask.append(int(1))
            else:
                negative_mask.append(int(0))

        for i in range(0, self.L):
            if positive_mask[i] == negative_mask[i]:
                pattern_mask.append(int(-1))
            else:
                pattern_mask.append(int(positive_mask[i]))

        return pattern_mask        
                           
    def mask_based_crossover(self):
        pattern_mask = self.mask()
        pf = 0.5

        for i in range(0, mating_pool_size - 1):
            if self.dice_throw(crossover_probability):
                # doing crossover operation
                parent1 = self.mating_pool[i]
                parent2 = self.mating_pool[i+1]

                ps = self.fittness_function(parent1)/(self.fittness_function(parent1) + self.fittness_function(parent2))

                child = []

                for j in range(0, self.L):
                    if parent1[j] == parent2[j] and parent1[j] == pattern_mask[j] :
                        child.append(parent1[j])

                    elif parent1[j] == parent2[j]  and pattern_mask[j] == -1 :
                        child.append(parent1[j])

                    elif parent1[j] == parent2[j]  and pattern_mask[j] != -1 :
                        if self.dice_throw(pf):
                            child.append(pattern_mask[j])
                        else :
                            child.append(parent1[j])

                    elif parent1[j] != parent2[j] and pattern_mask[j] == -1 :
                        if self.dice_throw(ps):
                            child.append(parent1[j])
                        else:
                            child.append(parent2[j])

                    elif parent1[j] != parent2[j]  and pattern_mask[j] != -1 :
                        if self.dice_throw(pf):
                            child.append(pattern_mask[j])
                        else :
                            if self.dice_throw(ps):
                                child.append(parent1[j])
                            else:
                                child.append(parent2[j])                             

                #print(child, len(child))
                if i == mating_pool_size - 1:
                    self.mating_pool[i] = child
                    self.mating_pool[i + 1] = child

                self.mating_pool[i] = child 
        pass                                  

    def three_parent_crossover(self):
        for i in range(0, mating_pool_size - 3, 3):
            if self.dice_throw(crossover_probability):
                # doing crossover operation
                parent1 = self.mating_pool[i]
                parent2 = self.mating_pool[i + 1]
                parent3 = self.mating_pool[i + 2]
                
                cut_points = random.sample(range(1, self.L - 1), 2)

                cut_point1 = min(cut_points)
                cut_point2 = max(cut_points)

                child1 = parent1[:cut_point1] + parent3[cut_point1:cut_point2] + parent2[cut_point2:]
                child2 = parent2[:cut_point1] + parent1[cut_point1:cut_point2] + parent3[cut_point2:]
                child3 = parent3[:cut_point1] + parent2[cut_point1:cut_point2] + parent1[cut_point2:]

                self.mating_pool[i] = child1
                self.mating_pool[i + 1] = child2;
                self.mating_pool[i + 2] = child3;
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
            #self.mask_based_crossover()
            #self.three_parent_crossover()
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
        return int(8);
    if string_val == "110":
        return int(7);
    if string_val == "101":
        return int(6);
    if string_val == "100":
        return int(5);
    if string_val == "011":
        return int(4);
    if string_val == "010":
        return int(3);
    if string_val == "001":
        return int(2);
    if string_val == "000":
        return int(1);

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
    return 1/(threats* 4 + 1) # + 1 is for division by zero escape 

def dev_fitness(input):
    # result is queens thar threating each other that must be minimum so we use 1/f(x)
    # of Length 24 showing 8 queens locations : three bits per one location
    # extracting real locations from coded locations
    locations = []
    
    for i in range(0, 24, 3):
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