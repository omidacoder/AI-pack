# Authors : Omid Davar , Mahdi Amiri Doomari
# Problem : Solve 8 Queens Using Real Genetic Algorithm
import math
import random
from matplotlib import pyplot as plt
# setting parameters
mating_pool_size = 14
mutation_probability = 0.01
crossover_probability = 0.8
stop_limit=1000
# node mapper shows neighbors. key is index of node and value is neighbors
node_mapper = [[2,5,6],[1,7,3],[2,8,4],[3,9,5],[1,10,4],[1,8,9],[2,9,10],[6,3,10],[6,4,7],[5,7,8]]
def round(input):
    if input - math.floor(input) > 0.5:
        return math.floor(input)+1
    return math.floor(input)
class real_genetic_algorithm:
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
                # generate random between 1 and 8
                generated = random.uniform(0,2)
                newRow.append(round(generated))
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
    # old crossover
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
                    # creating another random for this cell
                    self.mating_pool[i][j] = round(random.uniform(0, 2))
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
            if fit == 1:
                break
            self.means.append(fit)
            self.iterations.append(i)
            if fit > max:
                max = fit
            self.best_so_far.append(max)

def fitness(input):
    # we need to count collisions and calculate 1/f(x)
    colors = input
    collisions = 0
    for i in range(10):
        # each node has 3 neighbors
        for j in range(3):
            if input[i] == input[node_mapper[i][j] - 1]: # -1 becouse indices start from 0
                collisions += 1
    return 1/(collisions*4+1)
# using it for printing threats
def dev_fitness(input):
        # we need to count collisions and calculate 1/f(x)
    colors = input
    collisions = 0
    for i in range(10):
        # each node has 3 neighbors
        for j in range(3):
            if input[i] == input[node_mapper[i][j] - 1]:
                collisions += 1
    print('Collisions : ',collisions)
    return 1/(collisions*4+1)
# running all above
instance = real_genetic_algorithm(fitness,10)
instance.process()
print('Colorizing result population in GA is :')
for i in range(len(instance.population)):
    print('Population Member : ',instance.population[i])
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