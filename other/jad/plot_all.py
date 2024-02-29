import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# CSVs
df = pd.read_csv(data_path)

# Dataframe to numpy array
data_array = np.array(df)

n_samples = np.shape(data_array)[0]

# number of spectral frequencies bands
freq_bands = 184

spectral = data_array[:,0:freq_bands]
copy = spectral.copy()

# plotting
plt.figure(figsize=(16, 12))

print(f'CSV contains: {n_samples} samples')

for _, sample_signal in enumerate(copy):
    plt.plot(sample_signal)

#plt.title('Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()