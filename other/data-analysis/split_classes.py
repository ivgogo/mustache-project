import pandas as pd
import time

print("Reading data.csv")
csv_path = "/media/ivan/Ivan/my_data/data.csv"
df = pd.read_csv(csv_path)
print("Dataframe created!")

# labels
unique_labels = df.iloc[:, -1].unique()

for label in unique_labels:
    # filtering
    subset_df = df[df.iloc[:, -1] == label]

    print("New class starting...")
    time.sleep(3)
    print(f"Now working with class n: {label}")
    time.sleep(1)
    for index, row in subset_df.iterrows():
        print(row.iloc[-1])  

    output_path = f"/media/ivan/Ivan/my_data/class_n_{label}.csv"
    subset_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")