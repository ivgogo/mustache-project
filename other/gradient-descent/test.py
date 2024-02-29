import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd 
import numpy as np 
import cv2

# paths
df_false_positives = pd.read_csv("/media/ivan/Ivan/data_20_2_24/data_false_positives_production_day_first_condition.csv")
df_small_red = pd.read_csv("/media/ivan/Ivan/data_20_2_24/data_small_red.csv")
df_big_red_meat = pd.read_csv("/media/ivan/Ivan/data_20_2_24/data_big_red_on_meat.csv")
df_big_red_fat = pd.read_csv("/media/ivan/Ivan/data_20_2_24/data_big_red_on_fat.csv")

false_positives = np.array(df_false_positives)
small_red = np.array(df_small_red)
big_red_meat = np.array(df_big_red_meat)
big_red_fat = np.array(df_big_red_fat)
df_false_positives = df_false_positives.drop(df_false_positives.columns[0], axis='columns')    # axis=1 --> columns

print(df_big_red_meat)
# print(df_false_positives)

df_big_red_meat_m = df_big_red_meat.iloc[1000,:]
df_big_red_fat_m = df_big_red_fat.iloc[1000,:]

big_red_meat = np.array(df_big_red_meat_m)
big_red_fat = np.array(df_big_red_fat_m)

for _, sample_signal in enumerate(big_red_meat):
    print(sample_signal)
    plt.plot(sample_signal)

plt.title('Big red on meat')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()