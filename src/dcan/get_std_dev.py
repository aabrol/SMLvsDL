import pandas as pd

df = pd.read_csv('/home/miran045/reine097/projects/AlexNet_Abrol2021/data/ABCD/qc_with_paths.csv')
print(df.head())
print(df.std())
