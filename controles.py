from pathlib import Path

import pandas as pd

CSV = Path("data") / "marketing_AB.csv"   # le même nom que dans apercu.py
assert CSV.exists(), f"Fichier introuvable : {CSV}"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]

print("--- 1. taille des groupes ---")
tailles = df["test_group"].value_counts()
print(pd.DataFrame({"utilisateurs": tailles, "part_pct": (100 * tailles / len(df)).round(2)}))

print("\n--- 2. conversion par groupe ---")
conv = df.groupby("test_group")["converted"].agg(["count", "sum", "mean"])
conv.columns = ["utilisateurs", "convertis", "taux"]
conv["taux_pct"] = (100 * conv["taux"]).round(3)
print(conv[["utilisateurs", "convertis", "taux_pct"]])

print("\n--- 3. total_ads : extrêmes et quantiles, par groupe ---")
print(df.groupby("test_group")["total_ads"]
        .describe(percentiles=[0.5, 0.9, 0.99]).round(1))

print("\n--- 4. most_ads_hour : min et max ---")
print(df["most_ads_hour"].min(), df["most_ads_hour"].max())

print("\n--- 5. jour le plus exposé : part de chaque jour dans chaque groupe (%) ---")
jours = pd.crosstab(df["most_ads_day"], df["test_group"], normalize="columns") * 100
print(jours.round(1))

print("\n--- 6. heure moyenne par groupe ---")
print(df.groupby("test_group")["most_ads_hour"].mean().round(2))