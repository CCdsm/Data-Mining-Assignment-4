import pandas as pd
from sklearn.cluster import AgglomerativeClustering

col_names = ['mpg','cylinders','displacement','horsepower','weight',
             'acceleration','model_year','origin','car_name']
df = pd.read_csv('auto-mpg.data', sep=r'\s+', names=col_names, na_values='?')

continuous_cols = ['mpg','displacement','horsepower','weight','acceleration']
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
X = df[continuous_cols]

clusterer = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = clusterer.fit_predict(X)
df['cluster'] = labels

cluster_stats = df.groupby('cluster')[continuous_cols].agg(['mean','var'])
origin_stats  = df.groupby('origin')[continuous_cols].agg(['mean','var'])

print("=== Cluster statistics ===")
print(cluster_stats)
print("\n=== Origin statistics ===")
print(origin_stats)

print("\n=== Crosstab cluster vs origin ===")
print(pd.crosstab(df['cluster'], df['origin']))
