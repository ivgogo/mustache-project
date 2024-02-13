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
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_0.0.csv"     # 93818 x 553   #0  # dirty CB                                   # 45000
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_1.0.csv"     # 27084 x 553   #1  # Shadow CB                                 # 15000
#                                                                           #2  # Missclassified CB   
data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_3.0.csv"     # 44411 x 553   #3  # Meat                                      # 20000
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_4.0.csv"     # 17398 x 553   #4  # Meat Shadow                               # 8500
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_5.0.csv"     # 176 x 553     #5  # Missclassified meat                       # 80
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_6.0.csv"     # 2569 x 553    #6  # Fat                                       # 1250
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_7.0.csv"     # 10643 x 553   #7  # Fat Shadow                                # 5000
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_8.0.csv"     # 26436 x 553   #8  # Missclassified fat                        # 13000
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_9.0.csv"     # 152987 x 553  #9  # PEHD Red Plastic                          # 75000
#data_path = "/media/ivan/Ivan/classes_split_ivan/class_n_10.0.csv"    # 9476 x 553    #10  # PEHD small pieces Red plastic on PORK    # 4500

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

# all points of interest
points_list = [25,30,39,44,53,57,61,80,112,146,154,159]

n_total_samples = np.shape(data_array)[0]
num_samples_to_select = 20
random_samples_indexes = random.sample(range(n_total_samples), num_samples_to_select)
random_samples = spectral[random_samples_indexes, :]

# =========================================================================================

# Spectro de 184 valores
spectro = [0.1, 0.2, 0.3, ..., 0.9]

# Puntos de interés y sus números
puntos_interes = [(10, 'P1'), (30, 'P2'), (50, 'P3'), (100, 'P4')]

# Trazar el spectro
plt.plot(spectro)

# Agregar etiquetas para los puntos de interés
for punto in puntos_interes:
    plt.text(punto[0], spectro[punto[0]], punto[1], fontsize=8, ha='center', va='bottom')

for point in points_list:
    plt.text(point, )

# Etiquetas de ejes y título
plt.xlabel('Índice del valor')
plt.ylabel('Valor del espectro')
plt.title('Espectro con puntos de interés')

# Mostrar el gráfico
plt.show()
