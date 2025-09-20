from FiveRandCities import FiveRandCities
from SimulatedAnnealing import SimulatedAnnealing
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# ===============================
# Preparar ciudades y función de evaluación
# ===============================
cities = FiveRandCities(n_cities=0)

# Cargar matriz de distancias desde CSV
cities.load_distance_matrix("Optimization-Algorithms/Simulated Annealing/distances_asymmetric.csv")

def tsp_func(tour: np.ndarray) -> float:
    return FiveRandCities.tsp_func(tour, cities)

# ===============================
# Definir espacio de búsqueda
# ===============================
space = [
    Real(0.5, 10.0, name="T0"),        # temperatura inicial
    Real(0.60, 1.0, name="alpha"),      # factor de enfriamiento
    Integer(1, 1000, name="max_inner"),  # iteraciones internas
]

# ===============================
# Función objetivo para la búsqueda
# ===============================
@use_named_args(space)
def objective(T0, alpha, max_inner):
    sa = SimulatedAnnealing(
        func=tsp_func,
        x0=cities.random_tour(),
        T0=T0,
        alpha=alpha,
        max_iter=1000,     # fijo
        tol=1e-8,
        perturb_fn=FiveRandCities.perturb_2opt,
        cooling_fn=None
    )
    
    _, mejor_len = sa.optimizar(
        temp_min=1e-3,
        max_inner=max_inner,
        seed=None,       # sin semilla fija → explora estocasticidad
        verbose=False
    )
    return mejor_len  # minimizar longitud

# ===============================
# Ejecutar búsqueda bayesiana
# ===============================
res = gp_minimize(
    objective,
    space,
    n_calls=150,        # número de evaluaciones
    random_state=42,   # reproducibilidad
    n_initial_points=10
)

# ===============================
# Mostrar mejores parámetros
# ===============================
print("\n=== Mejores parámetros encontrados ===")
print(f"T0 = {res.x[0]}")
print(f"alpha = {res.x[1]}")
print(f"max_inner = {res.x[2]}")
print(f"Longitud mínima = {res.fun}")

# ===============================
# Ejecutar SA final con mejores parámetros
# ===============================
sa_final = SimulatedAnnealing(
    func=tsp_func,
    x0=cities.random_tour(),
    T0=res.x[0],
    alpha=res.x[1],
    max_iter=1000,
    tol=1e-8,
    perturb_fn=FiveRandCities.perturb_2opt,
    cooling_fn=None
)

mejor_tour, mejor_len = sa_final.optimizar(
    temp_min=1e-3,
    max_inner=res.x[2],
    seed=42,
    verbose=False
)

print("\n=== Resultado TSP Final ===")
print("Mejor tour encontrada:", mejor_tour)
print("Longitud del mejor tour:", mejor_len)

# ===============================
# Graficar convergencia
# ===============================  
hist_f = sa_final.historial_f
mejor_idx = np.argmin(hist_f)
mejor_val = hist_f[mejor_idx]

plt.figure(figsize=(8,5))
plt.plot(hist_f, label="Distancia del tour")
plt.scatter(mejor_idx, mejor_val, color='red', s=100, zorder=5, label=f'Mínimo: {mejor_val:.2f}')
plt.text(mejor_idx, mejor_val, f'{mejor_val:.2f}', color='red', fontsize=10,
         ha='left', va='bottom', fontweight='bold')
plt.xlabel("Iteración")
plt.ylabel("Distancia del tour")
plt.title("Convergencia de Simulated Annealing")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar figura
plt.savefig("convergencia_sa.png", dpi=300)
plt.show()