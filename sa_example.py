"""
Ejemplo de uso de la clase SimulatedAnnealing
para optimizar la función de Rosenbrock, Acley y Griewangk.
"""

import numpy as np
from SimulatedAnnealing import SimulatedAnnealing, perturb_uniform, cooling_log

import numpy as np

# ------------------------------
# Rosenbrock
# ------------------------------
def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Función de Rosenbrock en dimensión n.
    Mínimo global: f([1,1,...,1]) = 0

    f(x) = sum_{i=1}^{n-1} [ b*(x_{i+1} - x_i^2)^2 + (a - x_i)^2 ]

    Parámetros:
        x : np.ndarray
            Vector de entrada.
        a : float
            Valor objetivo para x_i.
        b : float
            Factor de penalización para el término cuadrático.

    Retorna:
        float: Valor de la función en x.
    """
    x = np.asarray(x)
    return np.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

# ------------------------------
# Ackley
# ------------------------------
def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi) -> float:
    """
    Función de Ackley en dimensión n.
    Mínimo global: f([0,0,...,0]) = 0

    f(x) = -a * exp(-b*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + exp(1)

    Parámetros:
        x : np.ndarray
            Vector de entrada.
        a, b, c : floats
            Parámetros estándar de Ackley.

    Retorna:
        float: Valor de la función en x.
    """
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e

# ------------------------------
# Griewangk
# ------------------------------
def griewangk(x: np.ndarray) -> float:
    """
    Función de Griewangk en dimensión n.
    Mínimo global: f([0,0,...,0]) = 0

    f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i)))

    Parámetros:
        x : np.ndarray
            Vector de entrada.

    Retorna:
        float: Valor de la función en x.
    """
    x = np.asarray(x)
    sum_sq = np.sum(x**2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return 1.0 + sum_sq - prod_cos

# Punto inicial aleatorio
x0 = np.random.uniform(-20, 10, size=5)

# Crear optimizador con funciones personalizadas
sa = SimulatedAnnealing(
    func=rosenbrock,
    x0=x0,
    T0=1000.0,
    alpha=0.01,
    max_iter=10000,
    tol=1e-8,
    perturb_fn=perturb_uniform,
    cooling_fn=cooling_log
)

# Ejecutar optimización
mejor_x, mejor_f = sa.optimizar(temp_min=1e-3, max_inner=200, seed=42, verbose=True)

print("\n=== Resultado ===")
print("Mejor solución encontrada:", mejor_x)
print("Valor en la mejor solución:", mejor_f)

# Graficar el recorrido proyectado en 2D
sa.graficar()