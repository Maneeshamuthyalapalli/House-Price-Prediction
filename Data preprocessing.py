print(df.describe())
df.ffill(inplace=True)
df.bfill(inplace=True)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
df.dropna(inplace=True)
print(df.columns)
import matplotlib.pyplot as plt
from scipy import stats
z_scores = stats.zscore(df['price'])
abs_z_scores = np.abs(z_scores)
filtered_df = df[abs_z_scores < 3]
plt.scatter(df['area'], df['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
df_outlier_removed = df.drop(filtered_df.index)
df_outlier_imputed = df.copy()
df_outlier_imputed['price'] = df_outlier_imputed['price'].fillna(df_outlier_imputed['price'].median())
