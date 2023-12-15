import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import random
import sys
import os

# number of spectral frequencies bands
r2 = 184

# Separated CSV file paths (hayat data)
data_path = "/media/ivan/Ivan/my_data/class_n_0.0.csv"     # 93818 x 553   #0  # dirty CB                                   # 45000
#data_path = "/media/ivan/Ivan/my_data/class_n_9.0.csv"     # 152987 x 553  #9  # PEHD Red Plastic                          # 75000
#data_path = "/media/ivan/Ivan/my_data/class_n_10.0.csv"    # 9476 x 553    #10  # PEHD small pieces Red plastic on PORK    # 4500

# Read csv
df = pd.read_csv(data_path)
data_array = np.array(df)
spectral = data_array[:,0:r2]

# conveyor belt indexes (Particular ones)
index_30 = 30
index_25 = 25

# plastic indexes
index_61 = 61
index_57 = 57
index_53 = 53

# moustache indexes
index_159 = 159
index_154 = 154
index_146 = 146

# other indexes
index_112 = 112
index_80 = 80
index_44 = 44
index_39 = 39

n_total_samples = np.shape(data_array)[0]
num_samples_to_select = 20
random_samples_indexes = random.sample(range(n_total_samples), num_samples_to_select)
random_samples = data_array[random_samples_indexes, :]

# plotting
'''
plt.figure(figsize=(16, 12))

for i, sample_signal in enumerate(random_samples):
    plt.plot(sample_signal, label=f'Sample n.{i + 1}')

plt.title('20 random signal samples with mean and std of points of interest')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
'''

# ====================================================================================================

medias_x = []
stds_x = []
medias_y = []
stds_y = []

points_list = [25,30,39,44,53,57,61,80,112,146,154,159]

for point in points_list:
    samples_in_point = df[(df['x'] == point[0]) & (df['y'] == point[1])]
    medias_x.append(samples_in_point['x'].mean())
    stds_x.append(samples_in_point['x'].std())
    medias_y.append(samples_in_point['y'].mean())
    stds_y.append(samples_in_point['y'].std())

# Convertir a numpy array para facilitar el manejo
medias_x = np.array(medias_x)
stds_x = np.array(stds_x)
medias_y = np.array(medias_y)
stds_y = np.array(stds_y)

# Tu gráfico original con las 20 muestras aleatorias
plt.scatter(df['x'], df['y'], label='Dataset Completo', alpha=0.5)

# Plotear cada punto del listado con la media y la desviación estándar
for i, punto in enumerate(points_list):
    plt.scatter(punto[0], punto[1], color='red', marker='x', label=f'Punto de Interés {i + 1}')
    plt.errorbar(punto[0], punto[1], xerr=stds_x[i], yerr=stds_y[i], color='red', linestyle='None', label='Desviación Estándar')

# Añadir etiquetas y leyenda
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.title('Dataset Completo con Media y Desviación Estándar en Puntos de Interés')

# Mostrar el gráfico
plt.show()






