# https://realpython.com/gradient-descent-algorithm-python/
# https://medium.com/@Coursesteach/machine-learning-part-17-gradient-descent-for-multiple-variables-1048c2ea5301

# Optimize
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

# Gradient descent
# https://scipy-lectures.org/advanced/mathematical_optimization/auto_examples/plot_gradient_descent.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html

import numpy as np
import pandas as pd
import scipy.optimize as opt

# Read data | data structure on the csv --> valor25,valor80,valor154
dataframe = pd.read_csv("/home/ivan/Descargas/csv_test.csv")
data_array = np.array(dataframe)

def f(x, data_array):
    
    # Number of samples (data)
    m = len(data_array)
    
    for i in range(len(data_array)):
        
        # Values of each point at each iteration
        value25, value80, value154 = data_array[i]

        # Operation 
        calc = (1/2*m)*(pow((value154*x[0]-value25),2)+pow((value154*x[1]-value80),2))

        return calc

# =============================
start = np.array([0, 0])

result = opt.minimize(f, start, args=(data_array,))
print(result)

# result_g = opt.fmin_cg(f, start, args=(data_array,))
# print(result_g)