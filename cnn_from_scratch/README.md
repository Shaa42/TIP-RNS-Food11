# CNN entraîné sans pré-entraînement

Petit réseau résiduel (8 blocs, largeurs 32-64-96-160), entraîné sur Food-11.

| | |
|---|---|
| Nombre de paramètres | 1 329 163 (1,33 M) |
| F1 macro (validation) | 0,847 |
| Taille des images | 128 × 128 |
| Séparation | 90 % entraînement / 10 % validation, stratifiée par classe (graine 42) |
| Entraînement | 40 époques, AdamW, OneCycle (LR max 2e-3), lot de 32, ~80 min sur RTX 3050 Laptop|

## Fichiers

- `train_cnn.py` : entraînement (données, modèle, boucle, évaluation)
- `predict.py` : prédiction sur le jeu de test → `runs/predictions.json`
- `plot_results.py` : figures du rapport
- `runs/` : résultats (journal par époque, matrice de confusion, figures, meilleur modèle, prédictions)

## Lancement

Modifier d'abord `DATA_DIR` (dans `train_cnn.py`) et `TEST_DIR` (dans `predict.py`) selon l'emplacement du jeu de données.

```bash
uv run python train_cnn.py
uv run python plot_results.py
uv run python predict.py
```
