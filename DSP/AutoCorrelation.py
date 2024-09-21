# Author : Omid Davar
# Subject : Computing Auto Correlation Of Random Data
# Student Number : 400155017
import numpy as np
# randomly chosen input
inp = [5,16,75,63,230,120,280,135,136]
# Mean
mean = np.mean(inp)
# Variance
var = np.var(inp)
# Normalized inp
ndata = inp - mean
auto_correlation = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
auto_correlation = auto_correlation / var / len(ndata)
print("input is :" , inp)
print("auto correlation is :" , auto_correlation)
