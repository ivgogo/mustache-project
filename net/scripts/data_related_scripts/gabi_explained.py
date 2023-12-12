from scipy import signal
import pandas as pd
import numpy as np
import sys
import os

# Script Gabi gave me, explained

# To work with Matlab files
# import scipy.io

'''
# last column in the dataframe is the label value: from 0.0 to 10.0
color_labels = {
    0.0:  (53, 136, 119),      # dirty CB
    1.0:  (153,136,119),       # Shadow CB 
    2.0:  (164,56,255),        # Missclassified CB
    3.0:  (223,116,255),       # Meat
    4.0:  (200,116,255),       # Meat Shadow
    5.0:  (0,116,255),         # Missclassified meat
    6.0:  (255,255,0),         # Fat
    7.0:  (200,255,0),         # Fat Shadow
    8.0:  (200,200,0),         # Missclassified fat
    9.0:  (76,76,76),          # PEHD Red plastic
    10.0: (100,76,76),         # PEHD small pieces Red plastic on PORK
    }

'''

# CSV file path
csv_file_path = "/home/ivan/Documentos/hyperspectral_project/data/red_big_part1_13_data.csv"

# Read csv
df = pd.read_csv(csv_file_path, index_col=0, header=None)

# Show dataframe
# print(df)

# Dataframe to numpy array
data_array = np.array(df)

# number of samples in the dataset (rows)
r1 = np.shape(data_array)[0]

# number of spectral frequencies bands
r2 = 184

# We have data for more than 184 frequencies bands
# so we select all rows and all columns from 0 till r2(184) with r2 not included so that means r2-1
spectral = data_array[:,0:r2]
# Print new size
print("spectral signal size: ", np.shape(spectral))

# We make a copy of the signal before filtering it
signal_filtered = spectral.copy()

'''
W is a mask (a Gaussian kernel) that you slide over the signal, and for each point, the filtered signal is the result of the convolution of signal with kernel w. 
In more simple words, for each element j of the 184 elements of signal, you compute: signal_filtered[j] = w[0]*signal[j-10] + w[1]*signal[j-10+1 ] +....+
w[19]*signal[j+10]. 
So each element of signal_filtered is a linear combination of its 20 neighbors with weights given by w.
'''
w = [0.0047, 0.0087, 0.0151, 0.0245, 0.0371, 0.0525, 0.0693, 0.0853, 0.0979, 0.1050, 0.1050, 0.0979, 0.0853, 0.0693, 0.0525, 0.0371, 0.0245, 0.0151, 0.0087, 0.0047]

# For every signal in signal_filtered  
for i in range(0, r1):
    # We apply lfilter --> lineal filter using lfilter from spicy.signal, we use w to apply the filter.
    # We overwrite the signal with the signal_filtered
    signal_filtered[i,:] = signal.lfilter(w, 1, spectral[i,:])

print("filtered spectral signal size: ", np.shape(signal_filtered))

# Another way of doing it gabi sent me
'''
for i in range(0, r1):
    signal_filtered[i,:] = signal.convolve(spectral[i,:], w, mode='same', method='direct')

print("filtered spectral signal size: ", np.shape(signal_filtered))
'''

# The first derivate tells us basically towards what direction the function is going so if we obtain
# the difference beetween the next point and the point before we'll get the direction towards the function is going.
signal_filtered_der1 = signal_filtered.copy()
for k in range(1, r2-1):
    signal_filtered_der1[:,k] = signal_filtered[:,k+1] -signal_filtered[:,k-1]

# The derivatives values are extremely small compared to the signal values, therefore we scale them by 1000 to be at in the same range as the signal that
# is in interval [0,1] and crop the values outside interval [-0.5,+0.5] 
signal_filtered_der1 = signal_filtered_der1 * 1000
signal_filtered_der1=np.where(signal_filtered_der1 < -0.5, -0.5, signal_filtered_der1)
signal_filtered_der1=np.where(signal_filtered_der1 > 0.5, 0.5, signal_filtered_der1)

# the same thing for 2nd derivative...
signal_filtered_der2 = signal_filtered.copy()
for k in range(1, r2-1):
    signal_filtered_der2[:,k] = signal_filtered[:,k+1] -2*signal_filtered[:,k]+signal_filtered[:,k-1]

signal_filtered_der2=signal_filtered_der2 * 1000
signal_filtered_der2=np.where(signal_filtered_der2 < -0.5, -0.5, signal_filtered_der2)
signal_filtered_der2=np.where(signal_filtered_der2 > 0.5, 0.5, signal_filtered_der2)

#save to panda
data_df = pd.DataFrame(signal_filtered)
panda_file_name = os.path.join("testing" + "_data.csv") 
data_df.to_csv(panda_file_name, header = False)
print('\ndata\n', data_df)
