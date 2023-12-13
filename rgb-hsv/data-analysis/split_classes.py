import pandas as pd

csv_path = "/media/ivan/Ivan/my_data/data.csv"
df = pd.read_csv(csv_path)

# labels
unique_labels = df.iloc[:, -1].unique()

for label in unique_labels:
    # filtering
    subset_df = df[df.iloc[:, -1] == label]

    output_path = f"/media/ivan/Ivan/my_data/class_n_{label}.csv"
    subset_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")