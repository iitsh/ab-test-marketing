from pathlib import Path

import pandas as pd

CSV = Path("data") / "marketing_AB.csv"
assert CSV.exists(), f"Fichier introuvable : {CSV}"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(CSV)

print("--- head ---")
print(df.head())
print("\n--- shape ---")    
print(df.shape)
print("\n--- dtypes ---")
print(df.dtypes)
print("\n--- valeurs manquantes ---")
print(df.isna().sum())
print("\n--- valeurs distinctes par colonne ---")
print(df.nunique())