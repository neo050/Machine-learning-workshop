import pandas as pd

dataset_url = "./Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States.csv"
df = pd.read_csv(dataset_url)

#Take a quick look at the first few rows of the dataset to get a sense of the data structure and the types of information contained within.
print(df.head())
