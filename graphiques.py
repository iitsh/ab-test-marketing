from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep

CSV = Path("data") / "marketing_AB.csv"   # le même nom que dans test_principal.py
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
    """Nombre avec d décimales et virgule."""
    return f"{x:.{d}f}".replace(".", ",")


def milliers(n):
    return f"{int(n):,}".replace(",", " ")


df = pd.read_csv(CSV).drop(columns="Unnamed: 0")
df.columns = ["user_id", "test_group", "converted", "total_ads", "most_ads_day", "most_ads_hour"]
df["est_ad"] = (df["test_group"] == "ad").astype(int)
df["conv"] = df["converted"].astype(int)

# ---------- Graphique 1 : conversion par groupe ----------
g = df.groupby("test_group")["conv"].agg(["sum", "count"])
xa, na = int(g.loc["ad", "sum"]), int(g.loc["ad", "count"])
xb, nb = int(g.loc["psa", "sum"]), int(g.loc["psa", "count"])
ecart_bas, ecart_haut = confint_proportions_2indep(xa, na, xb, nb, method="newcomb", compare="diff")
ecart = 100 * (xa / na - xb / nb)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for i, (nom, x, n) in enumerate([("psa", xb, nb), ("ad", xa, na)]):
    taux = 100 * x / n
    bas, haut = proportion_confint(x, n, method="wilson")
    ax.bar(i, taux, width=0.55, color=ACCENT if nom == "ad" else INK)
    ax.errorbar(i, taux, yerr=[[taux - 100 * bas], [100 * haut - taux]],
                color=MUTED, capsize=5, linewidth=1.6)
    ax.text(i, 100 * haut + 0.08, f"{fr(taux, 3)} %", ha="center", va="bottom", fontsize=11)
ax.text(0.5, 3.1, f"Écart : +{fr(ecart, 3)} point\nIC 95 % : {fr(100 * ecart_bas, 3)} à {fr(100 * ecart_haut, 3)}",
        ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"psa\n{milliers(nb)} utilisateurs", f"ad\n{milliers(na)} utilisateurs"],
                   color=INK)
ax.set_ylim(0, 3.7)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False, labelleft=False, bottom=False)
ax.set_title(f"Conversion : {fr(100 * xa / na, 3)} % dans le groupe ad, {fr(100 * xb / nb, 3)} % dans le groupe psa",
             loc="left", fontsize=13, pad=16)
fig.tight_layout()
fig.savefig(DOCS / "01_conversion_par_groupe.png", bbox_inches="tight")
plt.close(fig)

# ---------- Graphique 2 : écart par jour ----------
jours = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi",
         "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"}
lignes = []
for j, nom_fr in jours.items():
    d = df[df["most_ads_day"] == j]
    a, b = d[d["est_ad"] == 1], d[d["est_ad"] == 0]
    x1, n1, x0, n0 = int(a["conv"].sum()), len(a), int(b["conv"].sum()), len(b)
    bas, haut = confint_proportions_2indep(x1, n1, x0, n0, method="newcomb", compare="diff")
    lignes.append({"jour": nom_fr, "ecart": 100 * (x1 / n1 - x0 / n0),
                   "bas": 100 * bas, "haut": 100 * haut, "n_psa": n0})
par_jour = pd.DataFrame(lignes)
print(par_jour.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.8))
y = range(len(par_jour))
ax.hlines(y, par_jour["bas"], par_jour["haut"], color=MUTED, linewidth=2, zorder=1)
ax.plot(par_jour["ecart"], list(y), "o", color=INK, markersize=7, zorder=3)
ax.axvline(0, color=MUTED, linewidth=1, zorder=0)
ax.axvline(ecart, color=ACCENT, linewidth=1.4, zorder=2)
ax.text(ecart + 0.04, -0.75, f"écart global : +{fr(ecart)}", color=ACCENT, fontsize=9, va="center")
for i, r in par_jour.iterrows():
    ax.text(2.7, i, f"+{fr(r.ecart)}   ·   {milliers(r.n_psa)} utilisateurs psa",
            va="center", ha="left", fontsize=9)
ax.set_yticks(list(y))
ax.set_yticklabels(par_jour["jour"], color=INK)
ax.invert_yaxis()
ax.set_xlim(-1, 4.6)
ax.set_ylim(6.6, -1.1)
ax.set_xticks([-1, 0, 1, 2, 3])
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)
ax.set_xlabel("Écart de taux de conversion, ad moins psa (points de pourcentage), avec intervalle à 95 %",
              loc="left", fontsize=9)
ax.set_title("Le taux de conversion d'ad dépasse celui de psa chaque jour de la semaine",
             loc="left", fontsize=13, pad=16)
fig.tight_layout()
fig.savefig(DOCS / "02_ecart_par_jour.png", bbox_inches="tight")
plt.close(fig)

# ---------- Graphique 3 : conversion par heure ----------
h = df.groupby("most_ads_hour")["conv"].agg(["mean", "count"])
h["taux"] = 100 * h["mean"]
h_min, h_max = h["taux"].idxmin(), h["taux"].idxmax()
print(f"\nheure min : {h_min} h, {h.loc[h_min, 'taux']:.2f} % sur {h.loc[h_min, 'count']} utilisateurs")
print(f"heure max : {h_max} h, {h.loc[h_max, 'taux']:.2f} % sur {h.loc[h_max, 'count']} utilisateurs")
print(f"effectif par heure : min {h['count'].min()}, max {h['count'].max()}")

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.bar(h.index, h["taux"], color=INK, width=0.7)
ax.set_xticks(range(24))
ax.set_xticklabels([str(i) for i in range(24)], fontsize=8)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["0 %", "1 %", "2 %", "3 %"])
ax.set_xlabel("Heure de la journée où l'utilisateur a vu le plus de publicités (tous groupes confondus)",
              loc="left", fontsize=9)
ax.tick_params(left=False)
ax.spines["left"].set_visible(False)
ax.set_title(f"Le taux de conversion varie de {fr(h.loc[h_min, 'taux'])} % à {h_min} h "
             f"à {fr(h.loc[h_max, 'taux'])} % à {h_max} h",
             loc="left", fontsize=13, pad=16)
fig.tight_layout()
fig.savefig(DOCS / "03_conversion_par_heure.png", bbox_inches="tight")
plt.close(fig)

print("\nImages enregistrées dans docs/")