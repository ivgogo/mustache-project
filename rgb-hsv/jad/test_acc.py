import pandas as pd
import numpy as np
import sys
import os

data_path = "/media/ivan/Ivan/my_data/class_n_3.0.csv"      # class 3   # Meat                                      # 20000
#data_path = "/media/ivan/Ivan/my_data/class_n_10.0.csv"     # class 10  # PEHD small pieces Red plastic on PORK     # 4500

# Read csv
df = pd.read_csv(data_path)

data_array = np.array(df)

# number of samples in the dataset (rows)
r1 = np.shape(data_array)[0]

# number of spectral frequencies bands
r2 = 184

spectral = data_array[:,0:r2]

# important indexs to detect red small plastic
index_61 = 61
index_57 = 57

index_154 = 154
index_146 = 146

count = 0

for i in range(r1):
    result1 = spectral[i,index_61] - spectral[i,index_57]
    result2 = spectral[i,index_154] - spectral[i,index_146]
    if result1<=0 or result2<=0:
        count += 1

print(f'Total accuracy: {(count/r1)*100}')
