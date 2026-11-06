from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

CSV = Path("data") / "marketing_AB.csv"   # le même nom que dans graphiques.py
assert CSV.exists(), f"Fichier introuvable : {CSV}"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]
df["est_ad"] = (df["test_group"] == "ad").astype(int)
df["conv"] = df["converted"].astype(int)

print("--- 1. l'effet de ad varie-t-il selon le jour ou l'heure ? (test du rapport de vraisemblance) ---")
base = smf.logit("conv ~ est_ad + C(most_ads_day) + C(most_ads_hour)", data=df).fit(disp=0)
df["periode"] = pd.cut(df["most_ads_hour"], bins=[-1, 5, 11, 17, 23],
                       labels=["0-5 h", "6-11 h", "12-17 h", "18-23 h"])
base_periode = smf.logit("conv ~ est_ad + C(most_ads_day) + C(periode)", data=df).fit(disp=0)

tests = [
    ("jour           ", "conv ~ est_ad * C(most_ads_day) + C(most_ads_hour)", base),
    ("heure (24)     ", "conv ~ est_ad * C(most_ads_hour) + C(most_ads_day)", base),
    ("période (4)    ", "conv ~ est_ad * C(periode) + C(most_ads_day)", base_periode),
]
for nom, formule, reference in tests:
    complet = smf.logit(formule, data=df).fit(disp=0, maxiter=200)
    lr = 2 * (complet.llf - reference.llf)
    ddl = int(round(complet.df_model - reference.df_model))
    converge = "oui" if complet.mle_retvals["converged"] else "NON"
    print(f"interaction groupe x {nom}: khi2 = {lr:.1f}   ddl = {ddl}   "
          f"p = {stats.chi2.sf(lr, ddl):.3g}   convergence : {converge}")

print("\n--- 2. conversion par tranche de total_ads et par groupe ---")
bornes = [0, 4, 9, 24, 49, 99, np.inf]
noms = ["1 à 4", "5 à 9", "10 à 24", "25 à 49", "50 à 99", "100 et plus"]
df["tranche"] = pd.cut(df["total_ads"], bins=bornes, labels=noms)

lignes = []
for t in noms:
    d = df[df["tranche"] == t]
    a, b = d[d["est_ad"] == 1], d[d["est_ad"] == 0]
    x1, n1, x0, n0 = int(a["conv"].sum()), len(a), int(b["conv"].sum()), len(b)
    bas, haut = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff")
    lignes.append({"tranche": t, "n_ad": n1, "n_psa": n0,
                   "taux_ad_pct": 100 * x1 / n1, "taux_psa_pct": 100 * x0 / n0,
                   "ecart_points": 100 * (x1 / n1 - x0 / n0),
                   "ic_bas": 100 * bas, "ic_haut": 100 * haut})
print(pd.DataFrame(lignes).round(3).to_string(index=False))