from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

CSV = Path("data") / "marketing_AB.csv"   # le même nom que dans robustesse.py
assert CSV.exists(), f"Fichier introuvable : {CSV}"
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

PAPER, INK, MUTED, RULE, ACCENT = "#FAFAF7", "#1A1A1A", "#76756F", "#E5E4DF", "#D64B2A"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 11,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def milliers(n):
    return f"{int(n):,}".replace(",", " ")


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]
df["est_ad"] = (df["test_group"] == "ad").astype(int)
df["conv"] = df["converted"].astype(int)

noms = ["1 à 4", "5 à 9", "10 à 24", "25 à 49", "50 à 99", "100 et plus"]
df["tranche"] = pd.cut(df["total_ads"], bins=[0, 4, 9, 24, 49, 99, np.inf], labels=noms)


def comparer(d):
    a, b = d[d["est_ad"] == 1], d[d["est_ad"] == 0]
    x1, n1, x0, n0 = int(a["conv"].sum()), len(a), int(b["conv"].sum()), len(b)
    bas, haut = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff")
    return {"n_ad": n1, "n_psa": n0, "conv_ad": x1, "conv_psa": x0,
            "taux_ad": 100 * x1 / n1, "taux_psa": 100 * x0 / n0,
            "ecart": 100 * (x1 / n1 - x0 / n0), "bas": 100 * bas, "haut": 100 * haut}


print("--- 1. interaction groupe x tranche d'exposition ---")
base = smf.logit("conv ~ est_ad + C(tranche)", data=df).fit(disp=0)
complet = smf.logit("conv ~ est_ad * C(tranche)", data=df).fit(disp=0)
lr = 2 * (complet.llf - base.llf)
ddl = int(round(complet.df_model - base.df_model))
print(f"khi2 = {lr:.1f}   ddl = {ddl}   p = {stats.chi2.sf(lr, ddl):.3g}")

print("\n--- 2. écart sous et à partir de 25 publicités ---")
for nom, masque in [("moins de 25", df["total_ads"] < 25), ("25 et plus ", df["total_ads"] >= 25)]:
    r = comparer(df[masque])
    print(f"{nom} : ad {r['taux_ad']:.3f} % ({milliers(r['n_ad'])})   psa {r['taux_psa']:.3f} % "
          f"({milliers(r['n_psa'])})   écart {r['ecart']:+.3f} points   IC95 [{r['bas']:.3f} ; {r['haut']:.3f}]")

print("\n--- 3. poids des utilisateurs exposés à 25 publicités ou plus ---")
for nom, est in [("ad ", 1), ("psa", 0)]:
    g = df[df["est_ad"] == est]
    h = g["total_ads"] >= 25
    print(f"{nom} : {100 * h.mean():.1f} % des utilisateurs, "
          f"{100 * g.loc[h, 'conv'].sum() / g['conv'].sum():.1f} % des convertis")

print("\n--- 4. détail par tranche ---")
res = pd.DataFrame([{"tranche": t, **comparer(df[df["tranche"] == t])} for t in noms])
print(res.round(3).to_string(index=False))

# Garde-fou : le titre du graphique suppose que seules les 3 dernières tranches excluent 0 par le haut
positives = list(res.loc[res["bas"] > 0, "tranche"])
assert positives == noms[3:], f"Titre à revoir, tranches positives : {positives}"

fig, ax = plt.subplots(figsize=(10, 4.8))
y = list(range(len(res)))
ax.hlines(y, res["bas"], res["haut"], color=MUTED, linewidth=2, zorder=1)
ax.plot(res["ecart"], y, "o", color=INK, markersize=7, zorder=3)
ax.axvline(0, color=MUTED, linewidth=1, zorder=0)
ecart_global = 100 * (df.loc[df["est_ad"] == 1, "conv"].mean() - df.loc[df["est_ad"] == 0, "conv"].mean())
ax.axvline(ecart_global, color=ACCENT, linewidth=1.4, zorder=2)
ax.text(ecart_global + 0.1, -0.8, f"écart global : +{fr(ecart_global)}", color=ACCENT, fontsize=9, va="center")
for i, r in res.iterrows():
    ax.text(8.2, i,
            f"{fr(r.ecart):>6}   ·   ad {fr(r.taux_ad)} %   psa {fr(r.taux_psa)} %   ·   {milliers(r.n_psa)} utilisateurs psa",
            va="center", ha="left", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([f"{t} publicités" for t in res["tranche"]], color=INK)
ax.invert_yaxis()
ax.set_xlim(-1.5, 19)
ax.set_ylim(5.6, -1.2)
ax.set_xticks([0, 2, 4, 6, 8])
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)
ax.set_xlabel("Écart de taux de conversion, ad moins psa (points de pourcentage), avec intervalle à 95 %",
              loc="left", fontsize=9)
ax.set_title("L'écart est net à partir de 25 publicités vues, et absent en dessous",
             loc="left", fontsize=13, pad=16)
fig.tight_layout()
fig.savefig(DOCS / "04_ecart_par_exposition.png", bbox_inches="tight")
plt.close(fig)
print("\nImage enregistrée : docs/04_ecart_par_exposition.png")