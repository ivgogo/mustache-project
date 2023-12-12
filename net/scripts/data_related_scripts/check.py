import pandas as pd
import numpy as np
import time

start_time = time.time()

#data_path = "/media/ivan/Ivan/hyperspectral_project/data/conveyor_belt_part1_03_data.csv"
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_04_data.csv" # 49752 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_05_data.csv" # 78054 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/pork_part1_16_data.csv" # 117024 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_big_part1_13_data.csv" # 152986 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_small_on_fat_part2_09_data.csv" # 9475 x 554
#data_path = "/media/ivan/Ivan/hyperspectral_project/data/red_small_on_meat_part2_08_data.csv" # 5917 x 554

#data_path = "/home/ivan/Documentos/hyperspectral_project/data/red_small_on_meat_part2_08_data.csv"
#data_path = "/home/ivan/Documentos/hyperspectral_project/data/pork_part1_04_data.csv"
#data_path = "/home/ivan/Documentos/hyperspectral_project/combined_data.csv"

#data_path = "/home/ivan/Documentos/ivan/csv_test/pork_part1_04_data.csv"

#data_path = "/home/ivan/Documentos/hyperspectral_project/combined_data.csv"
#data_path = "/home/ivan/Documentos/ivan/data.csv"
#data_path = "/home/ivan/Documentos/ivan/ready/train/d2x/d2x.csv"

#data_path = "/home/ivan/Documentos/ivan/ready/sampled_data.csv"
#data_path = "/home/ivan/Documentos/ivan/ready/train/y/y.csv"
#data_path = "/home/ivan/Documentos/ivan_m/ready/sampled/sampled_data.csv"
data_path = "/home/ivan/Documentos/ivan_m/ready/train/d2x/d2x.csv"

df = pd.read_csv(data_path)
#selected_columns = 184
#df = df.iloc[:, 0:selected_columns]
#df = df.drop((df.columns[0]), axis=1)
print(df)

end_time = time.time()
execution_time = end_time - start_time
print(f"Elapsed time: {execution_time}")
