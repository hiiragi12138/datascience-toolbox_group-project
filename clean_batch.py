import pandas as pd

batch = pd.read_csv("batch.csv")

batch = batch.drop(columns=["Unnamed: 0"], errors='ignore')

batch["has_call_to_action"] = batch["has_call_to_action"].map({1: True, 0: False})
batch["post_datetime"] = pd.to_datetime(batch["post_datetime"])
batch["post_date"] = pd.to_datetime(batch["post_date"])

batch.to_csv("batch_cleaned.csv", index=False)
print("batch_cleaned.csv has been created successfully, number of rows: ", len(batch))
