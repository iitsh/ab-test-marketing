from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

CSV = Path("data") / "marketing_AB.csv"   
assert CSV.exists(), f"Fichier introuvable : {CSV}"

df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]
df["est_ad"] = (df["test_group"] == "ad").astype(int)
df["conv"] = df["converted"].astype(int)

ad = df[df["est_ad"] == 1]
psa = df[df["est_ad"] == 0]
n1, x1 = len(ad), int(ad["conv"].sum())
n0, x0 = len(psa), int(psa["conv"].sum())
p1, p0 = x1 / n1, x0 / n0

print("--- 1. test z sur deux proportions ---")
z, p = proportions_ztest([x1, x0], [n1, n0], alternative="two-sided")
print(f"ad  : {x1} / {n1} = {100 * p1:.3f} %")
print(f"psa : {x0} / {n0} = {100 * p0:.3f} %")
print(f"z = {z:.2f}   p = {p:.3g}")

print("\n--- 2. intervalles de confiance à 95 % ---")
bas, haut = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff")
print(f"écart ad - psa : {100 * (p1 - p0):.3f} points   IC95 [{100 * bas:.3f} ; {100 * haut:.3f}]")
bas_r, haut_r = confint_proportions_2indep(x1, n1, x0, n0, method="log", compare="ratio")
print(f"rapport ad / psa : {p1 / p0:.3f}   IC95 [{bas_r:.3f} ; {haut_r:.3f}]")
print(f"écart relatif : {100 * (p1 / p0 - 1):+.1f} %")

print("\n--- 3. régression logistique : odds ratio de ad ---")
brut = smf.logit("conv ~ est_ad", data=df).fit(disp=0)
ajuste = smf.logit("conv ~ est_ad + C(most_ads_day) + C(most_ads_hour)", data=df).fit(disp=0)
for nom, m in [("brut   ", brut), ("ajusté ", ajuste)]:
    ci = m.conf_int().loc["est_ad"]
    print(f"{nom}: OR = {np.exp(m.params['est_ad']):.3f}   "
          f"IC95 [{np.exp(ci[0]):.3f} ; {np.exp(ci[1]):.3f}]   p = {m.pvalues['est_ad']:.3g}")

print("\n--- 4. écart ajusté en points (standardisation) ---")
q1 = ajuste.predict(df.assign(est_ad=1)).mean()
q0 = ajuste.predict(df.assign(est_ad=0)).mean()
print(f"tous dans ad : {100 * q1:.3f} %   tous dans psa : {100 * q0:.3f} %   "
      f"écart : {100 * (q1 - q0):.3f} points")

print("\n--- 5. plus petit écart détectable (puissance 80 %, seuil 5 %) ---")
p_tot = (x1 + x0) / (n1 + n0)
se = np.sqrt(p_tot * (1 - p_tot) * (1 / n1 + 1 / n0))
print(f"erreur standard : {100 * se:.3f} points   écart détectable environ : {100 * 2.8 * se:.2f} points")