from FiveRandCities import FiveRandCities
from SimulatedAnnealing import SimulatedAnnealing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# Preparar ciudades y función de evaluación
# ===============================
cities = FiveRandCities(n_cities=0)

# Cargar matriz de distancias desde CSV
cities.load_distance_matrix("Optimization-Algorithms/Simulated Annealing/distances_asymmetric.csv")

def tsp_func(tour: np.ndarray) -> float:
    return FiveRandCities.tsp_func(tour, cities)

# ===============================
# Parámetros óptimos encontrados
# ===============================
T0_best = 1.0
alpha_best = 0.7346798407898818
max_inner_best = 478

# ===============================
# Ejecutar 1000 corridas de SA
# ===============================
resultados = []
mejor_tour_global = None
mejor_len_global = float("inf")

for run in range(1000):
    sa = SimulatedAnnealing(
        func=tsp_func,
        x0=cities.random_tour(),
        T0=T0_best,
        alpha=alpha_best,
        max_iter=1000,
        tol=1e-8,
        perturb_fn=FiveRandCities.perturb_2opt,
        cooling_fn=None
    )

    mejor_tour, mejor_len = sa.optimizar(
        temp_min=1e-3,
        max_inner=max_inner_best,
        seed=run,   # cada corrida con semilla distinta
        verbose=False
    )

    resultados.append(mejor_len)

    if mejor_len < mejor_len_global:
        mejor_len_global = mejor_len
        mejor_tour_global = mejor_tour

# ===============================
# Guardar resultados en CSV
# ===============================
df = pd.DataFrame({"Run": range(1, 1001), "Best_Length": resultados})
df.to_csv("resultados_sa.csv", index=False)

# ===============================
# Graficar resultados
# ===============================
plt.figure(figsize=(8,5))
plt.plot(resultados, marker="o", linestyle="--", alpha=0.7)
plt.axhline(np.mean(resultados), color="red", linestyle="--", label=f"Promedio: {np.mean(resultados):.2f}")
plt.scatter(np.argmin(resultados), mejor_len_global, color="green", s=100,
            label=f"Mejor: {mejor_len_global:.2f}")
plt.xlabel("Ejecución")
plt.ylabel("Longitud del tour")
plt.title("Resultados de 1000 ejecuciones de SA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados_sa.png", dpi=300)
plt.show()

# ===============================
# Imprimir mejor resultado global
# ===============================
print("\n=== Mejor resultado en 1000 ejecuciones ===")
print("Mejor tour encontrado:", mejor_tour_global)
print("Longitud del mejor tour:", mejor_len_global)
print(f"Promedio de longitudes: {np.mean(resultados):.4f}")
print(f"Desviación estándar: {np.std(resultados):.4f}")