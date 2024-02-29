import pandas as pd

# script created to split data and train with very small datasets in order to not blow up my laptop and to check if
# main code is working before uploading to the server

data_path = "/home/ivan/Documentos/ivan/data.csv"
df = pd.read_csv(data_path)

sample_size = 40000
small_df = df.sample(n=sample_size, random_state=42)
labels = small_df[small_df.columns[-1]]

changes = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3}
new_labels = labels.replace(changes)
small_df[small_df.columns[-1]] = new_labels
labels_u = small_df[small_df.columns[-1]].unique()

data_path_small = "/home/ivan/Documentos/ivan/small.csv"
small_df.to_csv(data_path_small, index=False)

print(f"New small.csv generated with {sample_size} random samples from data.csv")