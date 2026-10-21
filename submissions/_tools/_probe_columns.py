import os
import pandas as pd

DATA = "/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant/e2e_data/cpt_jyc_2026_e2e_bar1m"
p = os.path.join(DATA, "202311.0.feather")
df = pd.read_feather(p)
print("columns:", list(df.columns))
print("dtypes:")
print(df.dtypes)
print("head:")
print(df.head(3).to_string())
