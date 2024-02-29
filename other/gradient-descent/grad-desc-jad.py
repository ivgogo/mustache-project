import numpy as np
import pandas as pd


# Read data | data structure on the csv --> valor25,valor80,valor154
dataframe = pd.read_csv("/home/ivan/Descargas/csv_test.csv")
# dataframe = pd.read_csv("/home/ivan/Descargas/csv_test_dup.csv")
data_array = np.array(dataframe)

# https://pub.towardsai.net/gradient-descent-algorithm-explained-2fe9da0de9a2
def gradient_descent(data_array, lr):
    
    y1 = 0
    y2 = 0
    m = len(data_array)
    
    for i in range(len(data_array)):
        
        # Values of each point at each iteration
        value25, value80, value154 = data_array[i]

        # Operation 
        y1 = y1 - lr*((1/2*m)*(2*value154*y1-2*value154*value25))
        y2 = y2 - lr*((1/2*m)*(2*value154*y2-2*value154*value80))

    parameters = [y1, y2]
    return parameters

# start = np.array([0, 0])
g = gradient_descent(data_array, lr=0.00001)
print(f'Result: {g}')