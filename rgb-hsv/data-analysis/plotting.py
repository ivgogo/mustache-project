import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pandas as pd
import numpy as np

#data_path = "/media/ivan/Ivan/ivan/csv_test/conveyor_belt_part1_03_data.csv"        # 120902 x 553
data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part1_04_data.csv"                #  49753 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part2_05_data.csv"                #  40010 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/pork_part6_05_data.csv"                #  11870 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/red_big_part1_13_data.csv"             #  152987 x 553
#data_path = "/media/ivan/Ivan/ivan/csv_test/red_small_on_fat_part2_09_data.csv"    #  9476 x 553

df = pd.read_csv(data_path, header=None)
df = df.drop(df.columns[0], axis=1)
print(df)

data_array = np.array(df)

r1 = np.shape(data_array)[0]
r2 = 184

spectral = data_array[:,0:r2]
signal_filtered = spectral.copy()
w = [0.0047, 0.0087, 0.0151, 0.0245, 0.0371, 0.0525, 0.0693, 0.0853, 0.0979, 0.1050, 0.1050, 0.0979, 0.0853, 0.0693, 0.0525, 0.0371, 0.0245, 0.0151, 0.0087, 0.0047]

# to plot wihout filtering just comment these 2 lines
for i in range(0, r1):
    signal_filtered[i,:] = signal.lfilter(w, 1, spectral[i,:])

# random samples
#sample_signals = signal_filtered[1500:1520, :]
#sample_signals = signal_filtered[10500:10520, :]
sample_signals = signal_filtered[18500:18520, :]
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