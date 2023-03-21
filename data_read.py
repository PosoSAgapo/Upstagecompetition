import pandas as pd

df = pd.read_csv('data/train/purchase.csv')
df.to_csv('data/upstage/upstage.inter', sep='\t', index=None)