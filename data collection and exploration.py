import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Housing.csv')
print(df.head())
print(df.isnull().sum())
print(df.dtypes)
