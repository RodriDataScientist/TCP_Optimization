from FiveRandCities import FiveRandCities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# ===============================
# Preparar ciudades
# ===============================
cities = FiveRandCities(n_cities=0)
cities.load_distance_matrix("Optimization-Algorithms/Simulated Annealing/distances_asymmetric.csv")

# ===============================
# Tour ya encontrado (guardado)
# ===============================
tour_guardado = np.array([
    18, 19, 12, 13, 15, 11, 10,  6,  7,  9, 93, 89, 88, 86, 84, 83, 82, 81, 80, 79,
    78, 74, 72, 67, 70, 69, 71, 68, 66, 65, 59, 60, 61, 64, 55, 56, 57, 58, 54, 53,
    51, 49, 52, 48, 47, 42, 45, 46, 43, 39, 40, 41, 35, 36, 38, 37, 34, 33, 32, 31,
    29, 30, 25, 24, 23, 22, 21, 16, 14,  8,  5,  4,  3,  2,  1,  0, 99, 98, 96, 97,
    95, 94, 92, 91, 90, 87, 85, 77, 76, 75, 73, 62, 63, 50, 44, 28, 27, 26, 20, 17
])

# Evaluar longitud del tour guardado
longitud_guardada = FiveRandCities.tsp_func(tour_guardado, cities)
print("Longitud del tour guardado:", longitud_guardada)

# ===============================
# Leer CSV de resultados
# ===============================
df = pd.read_csv("Optimization-Algorithms/Simulated Annealing/resultados_sa.csv")
valores = df["Best_Length"].values

# ===============================
# Test de normalidad
# ===============================
stat, p_value = shapiro(valores)
print("\n=== Test de normalidad Shapiro-Wilk ===")
print(f"Statistic = {stat:.4f}, p-value = {p_value:.4f}")
if p_value > 0.05:
    print("No se rechaza H0 → la distribución parece normal.")
else:
    print("Se rechaza H0 → la distribución NO parece normal.")

# ===============================
# Ajuste normal y gráfico
# ===============================
mu, std = norm.fit(valores)

plt.figure(figsize=(8,5))
# Histograma de los datos
plt.hist(valores, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor="black", label="Resultados SA")

# Curva normal ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label=f'N(mu: {mu:.2f}, std: {std:.2f}²)')

# Punto del tour guardado
plt.axvline(longitud_guardada, color='green', linestyle="--", linewidth=2, label=f"Tour encontrado: {longitud_guardada:.2f}")

plt.title("Distribución de longitudes de SA (1000 ejecuciones)")
plt.xlabel("Longitud del tour")
plt.ylabel("Densidad")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("distribucion_sa.png", dpi=300)
plt.show()