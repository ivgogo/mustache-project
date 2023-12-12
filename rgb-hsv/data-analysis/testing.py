from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# CSV file path
#data_path = "/media/ivan/Ivan/ivan/csv_test/conveyor_belt_part1_03_data.csv"        # 120902 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part1_04_data.csv"                #  49753 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part2_05_data.csv"                #  40010 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part6_05_data.csv"                #  11870 x 553
data_path = "/media/ivan/Ivan/ivan/csv_test/red_big_part1_13_data.csv"             #  152987 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/red_small_on_fat_part2_09_data.csv"    #  9476 x 553

# Read csv
df = pd.read_csv(data_path, index_col=0, header=None)
max_position = df.max().max()
min_position = df.min().min()
print(max_position)
print(min_position)

# Show dataframe
print(df)

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
print("spectral signal size: ", np.shape(spectral))

# We make a copy of the signal before filtering it
signal_filtered = spectral.copy()


w = [0.0047, 0.0087, 0.0151, 0.0245, 0.0371, 0.0525, 0.0693, 0.0853, 0.0979, 0.1050, 0.1050, 0.0979, 0.0853, 0.0693, 0.0525, 0.0371, 0.0245, 0.0151, 0.0087, 0.0047]


for i in range(0, r1):
    signal_filtered[i,:] = signal.lfilter(w, 1, spectral[i,:])

print("filtered spectral signal size: ", np.shape(signal_filtered))


signal_filtered_der1 = signal_filtered.copy()
for k in range(1, r2-1):
    signal_filtered_der1[:,k] = signal_filtered[:,k+1] -signal_filtered[:,k-1]

signal_filtered_der1 = signal_filtered_der1 * 1000
signal_filtered_der1=np.where(signal_filtered_der1 < -0.5, -0.5, signal_filtered_der1)
signal_filtered_der1=np.where(signal_filtered_der1 > 0.5, 0.5, signal_filtered_der1)

new = pd.DataFrame(signal_filtered_der1)


sample_signals = spectral[18500:18510, :]
sample_signals_der = signal_filtered_der1[18500:18510, :]

# plotting
plt.figure(figsize=(16, 12))

for i, sample_signal in enumerate(sample_signals):
    plt.plot(sample_signal, label=f'Señal {i + 1}')

for i, sample_signals_der in enumerate(sample_signals_der):
    plt.plot(sample_signals_der, label=f'Señal {i + 1} derivado')

plt.title('Signals')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


