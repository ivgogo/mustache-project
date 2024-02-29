import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import sys
import os

# CSV file paths (hayat data)
#data_path = "/media/ivan/Ivan/ivan/csv_test/conveyor_belt_part1_03_data.csv"       # 120902 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part1_04_data.csv"                # 49753 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part2_05_data.csv"                # 40010 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part6_05_data.csv"                # 11870 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/red_big_part1_13_data.csv"             # 152987 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/red_small_on_fat_part2_09_data.csv"    # 9476 x 553

# Separated CSV file paths (hayat data)
#data_path = "/media/ivan/Ivan/my_data/class_n_0.0.csv"     # 93818 x 553   #0  # dirty CB                                   # 45000
#data_path = "/media/ivan/Ivan/my_data/class_n_1.0.csv"     # 27084 x 553   #1  # Shadow CB                                 # 15000
#                                                                           #2  # Missclassified CB   
data_path = "/media/ivan/Ivan/my_data/class_n_3.0.csv"     # 44411 x 553   #3  # Meat                                      # 20000
#data_path = "/media/ivan/Ivan/my_data/class_n_4.0.csv"     # 17398 x 553   #4  # Meat Shadow                               # 8500
#data_path = "/media/ivan/Ivan/my_data/class_n_5.0.csv"     # 176 x 553     #5  # Missclassified meat                       # 80
#data_path = "/media/ivan/Ivan/my_data/class_n_6.0.csv"     # 2569 x 553    #6  # Fat                                       # 1250
#data_path = "/media/ivan/Ivan/my_data/class_n_7.0.csv"     # 10643 x 553   #7  # Fat Shadow                                # 5000
#data_path = "/media/ivan/Ivan/my_data/class_n_8.0.csv"     # 26436 x 553   #8  # Missclassified fat                        # 13000
#data_path = "/media/ivan/Ivan/my_data/class_n_9.0.csv"     # 152987 x 553  #9  # PEHD Red Plastic                          # 75000
#data_path = "/media/ivan/Ivan/my_data/class_n_10.0.csv"    # 9476 x 553    #10  # PEHD small pieces Red plastic on PORK    # 4500

limit1 = 1250
mode = ""

# Read csv
df = pd.read_csv(data_path)
#print(df) # show dataframe

# Dataframe to numpy array
data_array = np.array(df)

# number of samples in the dataset (rows)
r1 = np.shape(data_array)[0]

# number of spectral frequencies bands
r2 = 184

# We have data for more than 184 frequencies bands
# so we select all rows and all columns from 0 till r2(184) with r2 not included so that means r2-1
spectral = data_array[:,0:r2]
#spectral = data_array
# Print new size

# We make a copy of the signal before filtering it
signal_filtered = spectral.copy()

w = [0.0047, 0.0087, 0.0151, 0.0245, 0.0371, 0.0525, 0.0693, 0.0853, 0.0979, 0.1050, 0.1050, 0.0979, 0.0853, 0.0693, 0.0525, 0.0371, 0.0245, 0.0151, 0.0087, 0.0047]
#w = [0.15, 0.25, 0.5, 0.25, 0.15]

for i in range(0, r1):
    signal_filtered[i,:] = signal.lfilter(w, 1, spectral[i,:])

# ======================= first derivative =======================
    
signal_filtered_der1 = signal_filtered.copy()
for k in range(1, r2-1):
    signal_filtered_der1[:,k] = signal_filtered[:,k+1] -signal_filtered[:,k-1]

signal_filtered_der1 = signal_filtered_der1 * 10
signal_filtered_der1=np.where(signal_filtered_der1 < -0.5, -0.5, signal_filtered_der1)
signal_filtered_der1=np.where(signal_filtered_der1 > 0.5, 0.5, signal_filtered_der1)

# ======================= second derivative =======================

signal_filtered_der2 = signal_filtered.copy()
for k in range(1, r2-1):
    signal_filtered_der2[:,k] = signal_filtered[:,k+1] -2*signal_filtered[:,k]+signal_filtered[:,k-1]

signal_filtered_der2=signal_filtered_der2 * 100
signal_filtered_der2=np.where(signal_filtered_der2 < -0.5, -0.5, signal_filtered_der2)
signal_filtered_der2=np.where(signal_filtered_der2 > 0.5, 0.5, signal_filtered_der2)

# ======================= plotting =======================

limit2 = limit1 + 20
sample_signals = spectral[limit1:limit2, :]
sample_signals_der1 = signal_filtered_der1[limit1:limit2, :]
sample_signals_der2 = signal_filtered_der2[limit1:limit2, :]

# plotting
plt.figure(figsize=(16, 12))

for i, sample_signal in enumerate(sample_signals):
    plt.plot(sample_signal, label=f'Signal n.{i + 1}')

#for i, sample_signals_der1 in enumerate(sample_signals_der1):
#    plt.plot(sample_signals_der1, label=f'Signal n.{i + 1} d1')

#for i, sample_signals_der2 in enumerate(sample_signals_der2):
#    plt.plot(sample_signals_der2, label=f'Signal n.{i + 1} d2')

plt.title('[20 random signal samples]')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


