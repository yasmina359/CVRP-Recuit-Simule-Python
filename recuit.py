import math
import os
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# 1. LECTURE DU FICHIER .vrp
def lire_vrp(chemin):
    coordonnees, demandes = [], []
    nom, capacite, section = "", None, None

    with open(chemin, "r") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            if ligne.startswith("NODE_COORD_SECTION"):
                section = "coord"
                continue
            if ligne.startswith("DEMAND_SECTION"):
                section = "demand"
                continue
            if ligne.startswith("DEPOT_SECTION") or ligne == "EOF":
                break

            if section == "coord":
                p = ligne.split()
                coordonnees.append((float(p[1]), float(p[2])))
            elif section == "demand":
                p = ligne.split()
                demandes.append(int(p[1]))
            elif ":" in ligne:
                cle, val = [s.strip() for s in ligne.split(":", 1)]
                if cle == "NAME":
                    nom = val
                elif cle == "CAPACITY":
                    capacite = int(val)

    m = re.search(r"-k(\d+)", nom)          
    k = int(m.group(1)) if m else None
    return nom, coordonnees, demandes, capacite, k


file_path = input("Nom du fichier CVRPLIB (.vrp) [A-n45-k7.vrp] : ").strip() or "A-n45-k7.vrp"
if not os.path.exists(file_path):
    sys.exit("Le fichier n'existe pas ! Vérifiez qu'il est dans le même dossier que ce script.")

nom, coordonnees, demandes, capacite_vehicule, nombre_vehicules = lire_vrp(file_path)

if capacite_vehicule is None:
    capacite_vehicule = int(input("Capacité du véhicule : "))
if nombre_vehicules is None:
    nombre_vehicules = int(input("Nombre de véhicules : "))

sol_opt_input = input("Solution optimale connue (vide si inconnue) [1146 pour A-n45-k7] : ").strip()
solution_opt = float(sol_opt_input) if sol_opt_input else None

nb_noeuds = len(coordonnees)
nb_clients = nb_noeuds - 1

print(f"\nInstance : {nom} | clients : {nb_clients} | K = {nombre_vehicules} | Q = {capacite_vehicule}")
print(f"Demande totale = {sum(demandes)} -> minimum {math.ceil(sum(demandes) / capacite_vehicule)} véhicules\n")

# 2. MATRICE DES DISTANCES (liste de listes : accès rapide en Python)
xy = np.array(coordonnees)
matrice_distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)).tolist()

# 3. FONCTION D'ÉVALUATION (Split + pénalité)
LAMBDA = 1000
def decouper(solution):
    routes, route, charge = [], [0], 0
    for c in solution:
        if charge + demandes[c] > capacite_vehicule:
            route.append(0)
            routes.append(route)
            route, charge = [0], 0
        route.append(c)
        charge += demandes[c]
    route.append(0)
    routes.append(route)
    return routes


def distance_route(route):
    return sum(matrice_distances[route[i]][route[i + 1]] for i in range(len(route) - 1))


def fitness(solution):
    routes = decouper(solution)
    distance = sum(distance_route(r) for r in routes)
    penalite = LAMBDA * max(0, len(routes) - nombre_vehicules)
    return distance + penalite, distance, routes
# 4. VOISINAGE : Swap (20 %), Shift (40 %), 2-opt (40 %)
P_SWAP, P_SHIFT = 0.20, 0.40      # le reste (0.40) = 2-opt
def generer_voisin(solution):
    voisin = solution[:]
    n = len(voisin)
    choix = random.random()

    if choix < P_SWAP:                               # Swap
        i, j = random.sample(range(n), 2)
        voisin[i], voisin[j] = voisin[j], voisin[i]
    elif choix < P_SWAP + P_SHIFT:                   # Shift
        i = random.randrange(n)
        client = voisin.pop(i)
        voisin.insert(random.randrange(n), client)
    else:                                            # 2-opt
        i, j = sorted(random.sample(range(n), 2))
        voisin[i:j + 1] = voisin[i:j + 1][::-1]
    return voisin

# 5. RECUIT SIMULÉ
def recuit_simule(T0=100.0, T_finale=0.01, alpha=0.995, iterations_par_palier=200, seed=None):
    if seed is not None:
        random.seed(seed)

    courante = list(range(1, nb_clients + 1))
    random.shuffle(courante)                          # solution initiale aléatoire
    cout_courant, _, _ = fitness(courante)

    meilleure, meilleur_cout = courante[:], cout_courant
    historique_best, historique_cur = [], []
    T = T0

    while T > T_finale:
        for _ in range(iterations_par_palier):
            voisin = generer_voisin(courante)
            cout_voisin, _, _ = fitness(voisin)
            delta = cout_voisin - cout_courant

            # Règle de Metropolis
            if delta <= 0 or random.random() < math.exp(-delta / T):
                courante, cout_courant = voisin, cout_voisin
                if cout_courant < meilleur_cout:
                    meilleure, meilleur_cout = courante[:], cout_courant

        historique_best.append(meilleur_cout)
        historique_cur.append(cout_courant)
        T *= alpha

    return meilleure, meilleur_cout, historique_best, historique_cur

# 6. EXÉCUTIONS MULTIPLES
NB_EXECUTIONS = 10
PARAMS = dict(T0=100.0, T_finale=0.01, alpha=0.995, iterations_par_palier=200)
print("Paramètres :", PARAMS, "| Swap/Shift/2-opt = 20/40/40 %\n")

resultats = []
global_best, global_best_cost, global_hist = None, float("inf"), None

for run in range(1, NB_EXECUTIONS + 1):
    debut = time.time()
    sol, cout, h_best, h_cur = recuit_simule(seed=run, **PARAMS)
    duree = time.time() - debut
    _, dist, routes = fitness(sol)
    ecart = (dist - solution_opt) / solution_opt * 100 if solution_opt else float("nan")
    resultats.append((cout, dist, len(routes), ecart, duree))
    print(f"Exécution {run:2d} | distance = {dist:8.2f} | véhicules = {len(routes)} "
          f"| écart = {ecart:6.2f} % | temps = {duree:5.1f} s")
    if cout < global_best_cost:
        global_best, global_best_cost, global_hist = sol, cout, (h_best, h_cur)

dists = [r[1] for r in resultats]
print("\n--- Statistiques sur", NB_EXECUTIONS, "exécutions ---")
print(f"Meilleure distance : {min(dists):.2f}")
print(f"Distance moyenne   : {np.mean(dists):.2f}")
print(f"Pire distance      : {max(dists):.2f}")
print(f"Écart-type         : {np.std(dists):.2f}")
print(f"Temps moyen        : {np.mean([r[4] for r in resultats]):.1f} s")
if solution_opt:
    print(f"Écart moyen / optimum : {(np.mean(dists) - solution_opt) / solution_opt * 100:.2f} %")
    print(f"Écart meilleur / optimum : {(min(dists) - solution_opt) / solution_opt * 100:.2f} %")

# 7. MEILLEURE SOLUTION
_, distance_meilleure, routes_meilleures = fitness(global_best)
print("\nMeilleure solution :")
for i, r in enumerate(routes_meilleures, 1):
    charge = sum(demandes[c] for c in r)
    print(f"  Véhicule {i}: {r} | charge = {charge}/{capacite_vehicule} | distance = {distance_route(r):.2f}")
print(f"Distance totale : {distance_meilleure:.2f} | véhicules utilisés : {len(routes_meilleures)}")
if len(routes_meilleures) > nombre_vehicules:
    print("ATTENTION : solution non faisable (trop de véhicules).")

# 8. GRAPHIQUES (enregistrés dans des fichiers ET affichés)
cmap = plt.get_cmap("tab10")
plt.figure(figsize=(9, 7))
for idx, route in enumerate(routes_meilleures):
    x = [coordonnees[n][0] for n in route]
    y = [coordonnees[n][1] for n in route]
    charge = sum(demandes[c] for c in route)
    plt.plot(x, y, marker="o", markersize=4, color=cmap(idx % 10),
             label=f"Véhicule {idx + 1} (charge {charge})")
    for n in route:
        if n != 0:
            plt.text(coordonnees[n][0] + 0.5, coordonnees[n][1] + 0.5, str(n), fontsize=7)
plt.scatter(*coordonnees[0], c="k", s=100, marker="s", zorder=5, label="Dépôt")
plt.title(f"Visualisation du CVRP par Recuit Simulé ({nom}) - distance = {distance_meilleure:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tournees.png", dpi=150)

plt.figure(figsize=(9, 5))
plt.plot(global_hist[1], alpha=0.5, label="Solution courante")
plt.plot(global_hist[0], "b-", linewidth=2, label="Meilleure solution")
if solution_opt:
    plt.axhline(solution_opt, color="g", linestyle="--", label=f"Optimum ({solution_opt:.0f})")
plt.ylim(top=min(max(global_hist[1]), 3000))
plt.title("Évolution du coût par palier de température")
plt.xlabel("Paliers de température")
plt.ylabel("Coût f(S)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
print("\nFigures enregistrées : tournees.png, convergence.png")
plt.show()