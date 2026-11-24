# Résolution du CVRP par Recuit Simulé

Ce projet implémente la métaheuristique du **recuit simulé** en Python pour résoudre le problème de tournées de véhicules avec contraintes de capacité (**CVRP**) sur l'instance `A-n45-k7.vrp`.

## Structure du projet
- `recuit.py` : Code principal de l'algorithme du recuit simulé.
- `A-n45-k7.vrp` : Fichier de données de l'instance testée.
- `requirements.txt` : Bibliothèques Python nécessaires.
- `resultats/` : Graphiques des tournées obtenues et de la courbe de convergence.

## Exécution
1. Installer les dépendances :
   `pip install -r requirements.txt`
2. Lancer le script :
   `python recuit.py`