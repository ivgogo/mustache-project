import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import os

# Separated CSV file paths (hayat data)
data_path_conveyor = "/media/ivan/Ivan/my_data/class_n_0.0.csv"     # 93818 x 553   #0  # dirty CB                                   # 45000                                                                       #2  # Missclassified CB   
data_path_meat = "/media/ivan/Ivan/my_data/class_n_3.0.csv"     # 44411 x 553   #3  # Meat                                      # 20000
data_path_fat = "/media/ivan/Ivan/my_data/class_n_6.0.csv"     # 2569 x 553    #6  # Fat                                       # 1250
data_path_red_big = "/media/ivan/Ivan/my_data/class_n_9.0.csv"     # 152987 x 553  #9  # PEHD Red Plastic                          # 75000
data_path_red_small = "/media/ivan/Ivan/my_data/class_n_10.0.csv"    # 9476 x 553    #10  # PEHD small pieces Red plastic on PORK    # 4500


# Read csv
df = pd.read_csv(data_path_fat)
df_spectral = df.iloc[:, :184]

df_spectral['mean'] = df_spectral.mean(axis=1)

min_mean = df_spectral.loc[df_spectral['mean'].idxmin()]
pos = df_spectral["mean"].idxmin()
print(f'Position: {pos}')

plt.plot(min_mean, label=".")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mínim')
plt.legend()
plt.show()


