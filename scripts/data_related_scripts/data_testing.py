import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pandas as pd
import numpy as np
import time
import sys
import os

#csv_file_path = "/home/ivan/Documentos/hyperspectral_project/data/red_big_part1_13_data.csv" # 152986 x 554

#data_path = "/media/ivan/Ivan/hyperspectral_project/data/conveyor_belt_part1_03_data.csv"
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_04_data.csv" # 49752 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_05_data.csv" # 78054 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_16_data.csv" # 117024 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_big_part1_13_data.csv" # 152986 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_small_on_fat_part2_09_data.csv" # 9475 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_small_on_meat_part2_08_data.csv" # 5917 x 554

data_path = "/home/ivan/Documentos/ivan_m/data.csv/"

df = pd.read_csv(data_path, index_col=0, header=None)
data_array = np.array(df)

r1 = np.shape(data_array)[0]
r2 = 184
#spectral = data_array[:,0:r2]
spectral = data_array[:,141:165]
signal_filtered = spectral.copy()
w = [0.0047, 0.0087, 0.0151, 0.0245, 0.0371, 0.0525, 0.0693, 0.0853, 0.0979, 0.1050, 0.1050, 0.0979, 0.0853, 0.0693, 0.0525, 0.0371, 0.0245, 0.0151, 0.0087, 0.0047]

# to plot wihout filtering just comment these 2 lines
#for i in range(0, r1):
#    signal_filtered[i,:] = signal.lfilter(w, 1, spectral[i,:])

# random samples
sample_signals = signal_filtered[1500:1520, :]
#sample_signals = signal_filtered[10500:10520, :]
#sample_signals = signal_filtered[18500:18520, :]
#sample_signals = signal_filtered[100000:100020, :]
#sample_signals = signal_filtered[150000:150020, :]

# plotting
plt.figure(figsize=(16, 12))

for i, sample_signal in enumerate(sample_signals):
    plt.plot(sample_signal, label=f'Señal {i + 1}')

plt.title('Signals')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()