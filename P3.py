import pandas as pd
from sklearn.datasets import load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score

try:
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    print("使用 load_wine() 载入数据")
except Exception as e:
    print(f"load_wine() 出错：{e}\n改用 OpenML")
    wine = fetch_openml(name="wine", as_frame=True)
    X = wine.data
    y = wine.target.astype(int)
print(f"数据维度: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
homo  = homogeneity_score(y, labels)
compl = completeness_score(y, labels)
print(f"\nHomogeneity Score   : {homo:.4f}")
print(f"Completeness Score  : {compl:.4f}")