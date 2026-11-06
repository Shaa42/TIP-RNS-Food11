"""
Figures pour le rapport, à partir de runs/log.csv et runs/confusion.csv.

Lancement (après l'entraînement) :
    .venv\\Scripts\\python.exe plot_results.py

Sorties (dans runs/) :
    courbes_perte.png        perte d'entraînement et de validation par époque
    courbes_f1.png           exactitude et F1 macro de validation par époque
    matrice_confusion.png    matrice de confusion normalisée (meilleure époque)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # pas de fenêtre : on enregistre directement les images
import matplotlib.pyplot as plt

RUN = Path("runs")

# --- courbes ---
rows = list(csv.DictReader(open(RUN / "log.csv", encoding="utf-8")))
ep = [int(r["epoch"]) for r in rows]
col = lambda k: [float(r[k]) for r in rows]

plt.figure(figsize=(6, 4))
plt.plot(ep, col("train_loss"), label="entraînement")
plt.plot(ep, col("val_loss"), label="validation")
plt.xlabel("époque"); plt.ylabel("perte (entropie croisée)"); plt.legend(); plt.grid(alpha=.3)
plt.title("Courbes de perte")
plt.tight_layout(); plt.savefig(RUN / "courbes_perte.png", dpi=200); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(ep, col("val_acc"), label="exactitude (validation)")
plt.plot(ep, col("val_macro_f1"), label="F1 macro (validation)")
plt.xlabel("époque"); plt.ylabel("score"); plt.legend(); plt.grid(alpha=.3)
plt.title("Performance en validation")
plt.tight_layout(); plt.savefig(RUN / "courbes_f1.png", dpi=200); plt.close()

# --- matrice de confusion (chaque ligne divisée par son total = rappel par classe sur la diagonale) ---
lines = list(csv.reader(open(RUN / "confusion.csv", encoding="utf-8")))
classes = lines[0][1:]
mat = [[int(v) for v in row[1:]] for row in lines[1:]]
norm = [[v / max(1, sum(row)) for v in row] for row in mat]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("classe prédite"); ax.set_ylabel("vraie classe")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if norm[i][j] > 0.5 else "black")
fig.colorbar(im)
ax.set_title("Matrice de confusion normalisée (validation)")
fig.tight_layout(); fig.savefig(RUN / "matrice_confusion.png", dpi=200); plt.close(fig)

print("Figures enregistrées dans", RUN)
