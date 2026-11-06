from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path("data") / "marketing_AB.csv"  
assert CSV.exists(), f"Fichier introuvable : {CSV}"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]


def khi2(variable):
    """Khi-deux d'indépendance entre la variable et le groupe, avec le V de Cramér."""
    tableau = pd.crosstab(df[variable], df["test_group"])
    chi2, p, ddl, _ = stats.chi2_contingency(tableau)
    v = np.sqrt(chi2 / tableau.to_numpy().sum())   # 2 groupes : min(lignes, colonnes) - 1 = 1
    return chi2, p, ddl, v


print("--- 1. jour : répartition identique dans les deux groupes ? ---")
chi2, p, ddl, v = khi2("most_ads_day")
print(f"khi2 = {chi2:.1f}   ddl = {ddl}   p = {p:.3g}   V de Cramer = {v:.4f}")

print("\n--- 2. heure : répartition identique dans les deux groupes ? ---")
chi2, p, ddl, v = khi2("most_ads_hour")
print(f"khi2 = {chi2:.1f}   ddl = {ddl}   p = {p:.3g}   V de Cramer = {v:.4f}")

print("\n--- 3. total_ads : Mann-Whitney entre les deux groupes ---")
a = df.loc[df["test_group"] == "ad", "total_ads"]
b = df.loc[df["test_group"] == "psa", "total_ads"]
u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"U = {u:.4g}   p = {p:.3g}   médiane ad = {a.median()}   médiane psa = {b.median()}")

print("\n--- 4. conversion par jour et par groupe (%) ---")
ordre = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
par_jour = (df.pivot_table(index="most_ads_day", columns="test_group",
                           values="converted", aggfunc="mean") * 100).reindex(ordre)
par_jour["ecart_points"] = par_jour["ad"] - par_jour["psa"]
par_jour["psa_utilisateurs"] = (df[df["test_group"] == "psa"]["most_ads_day"]
                                .value_counts().reindex(ordre))
print(par_jour.round(3))

print("\n--- 5. conversion par heure, tous groupes (%) ---")
par_heure = (df.groupby("most_ads_hour")["converted"].mean() * 100).round(2)
print(par_heure.to_string())