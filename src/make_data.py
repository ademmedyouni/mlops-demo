from sklearn.datasets import load_wine

df = load_wine(as_frame=True).frame
df.to_csv("data/raw/wine.csv", index=False)