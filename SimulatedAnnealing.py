"""
Módulo: simulated_annealing
---------------------------
Implementación flexible del algoritmo de Recocido Simulado (Simulated Annealing).

Características:
- Permite definir funciones personalizadas de perturbación del vecindario.
- Permite definir funciones personalizadas de enfriamiento de la temperatura.
- Guarda historial de soluciones, valores y temperaturas.
- Incluye método para graficar el recorrido proyectado en 2D mediante PCA.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Callable, Tuple, Union, Optional

ArrayLike = Union[float, np.ndarray, list, tuple]

class SimulatedAnnealing:
    """
    Clase que implementa el algoritmo de Recocido Simulado para minimizar funciones arbitrarias.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        x0: ArrayLike,
        T0: float = 100.0,
        alpha: float = 0.95,
        max_iter: int = 10000,
        tol: float = 1e-6,
        perturb_fn: Optional[Callable] = None,
        cooling_fn: Optional[Callable] = None,
    ):
        """
        Constructor de la clase SimulatedAnnealing.

        Parámetros:
            func       : función objetivo a minimizar. Debe recibir un np.ndarray y devolver un float.
            x0         : punto inicial (float, lista, tupla o array).
            T0         : temperatura inicial.
            alpha      : factor de enfriamiento (usado si se emplea el enfriamiento por defecto).
            max_iter   : número máximo de iteraciones totales.
            tol        : tolerancia mínima para considerar una mejora significativa.
            perturb_fn : función de perturbación personalizada (x, T, T0) -> nuevo_x.
                         Si no se especifica, se usa perturbación gaussiana.
            cooling_fn : función de enfriamiento personalizada (T, alpha) -> nuevo_T.
                         Si no se especifica, se usa enfriamiento geométrico.
        """
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        if self.x0.ndim == 0:
            self.x0 = np.array([self.x0], dtype=float)

        self.T0 = float(T0)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Función de perturbación por defecto: gaussiana
        if perturb_fn is None:
            def default_perturb(x, T, T0):
                escala = np.sqrt(T / T0)
                return x + np.random.normal(0, escala, size=x.shape)
            self.perturb_fn = default_perturb
        else:
            self.perturb_fn = perturb_fn

        # Función de enfriamiento por defecto: geométrica
        if cooling_fn is None:
            def default_cooling(T, alpha):
                return max(T * alpha, 1e-12)
            self.cooling_fn = default_cooling
        else:
            self.cooling_fn = cooling_fn

        # Estado inicial
        self.x = self.x0.copy()
        self.fx = float(self.func(self.x))
        self.T = self.T0
        self.iter = 0

        # Historial
        self.historial_x = [self.x.copy()]
        self.historial_f = [self.fx]
        self.historial_T = [self.T]

        # Mejor solución encontrada
        self.best_x = self.x.copy()
        self.best_f = self.fx

    def generar_vecino(self, x: np.ndarray) -> np.ndarray:
        """
        Genera una solución vecina usando la función de perturbación configurada.

        Parámetros:
            x : solución actual.

        Retorna:
            np.ndarray con la nueva solución candidata.
        """
        return self.perturb_fn(x, self.T, self.T0)

    def enfriar_temperatura(self) -> float:
        """
        Actualiza la temperatura usando la función de enfriamiento configurada.

        Retorna:
            float con el nuevo valor de la temperatura.
        """
        self.T = self.cooling_fn(self.T, self.alpha)
        return self.T

    def probabilidad_aceptacion(self, delta_f: float, T: float) -> float:
        """
        Calcula la probabilidad de aceptar una solución peor.

        Parámetros:
            delta_f : diferencia f_nuevo - f_actual.
            T       : temperatura actual.

        Retorna:
            Probabilidad de aceptación entre 0 y 1.
        """
        if delta_f < 0:
            return 1.0
        if T <= 0:
            return 0.0
        try:
            return math.exp(-delta_f / T)
        except OverflowError:
            return 0.0

    def optimizar(
        self,
        temp_min: float = 1e-3,
        max_inner: int = 100,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el algoritmo de Recocido Simulado.

        Parámetros:
            temp_min  : temperatura mínima para detener el algoritmo.
            max_inner : número de vecinos a probar por cada nivel de temperatura.
            seed      : semilla opcional para reproducibilidad.
            verbose   : si True, imprime información de progreso.

        Retorna:
            (mejor_solución, valor_función_en_mejor_solución)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reinicializar estado
        self.T = self.T0
        self.x = self.x0.copy()
        self.fx = float(self.func(self.x))
        self.best_x = self.x.copy()
        self.best_f = self.fx
        self.iter = 0

        while self.T > temp_min and self.iter < self.max_iter:
            for _ in range(max_inner):
                if self.iter >= self.max_iter:
                    break

                vecino = self.generar_vecino(self.x)
                f_vecino = float(self.func(vecino))
                delta = f_vecino - self.fx

                if delta < 0 or random.random() < self.probabilidad_aceptacion(delta, self.T):
                    self.x = vecino
                    self.fx = f_vecino

                    if self.fx < self.best_f - self.tol:
                        self.best_x = self.x.copy()
                        self.best_f = self.fx

                # Guardar historial
                self.historial_x.append(self.x.copy())
                self.historial_f.append(self.fx)
                self.historial_T.append(self.T)

                self.iter += 1

            prev_T = self.T
            self.enfriar_temperatura()
            if verbose:
                print(f"iter={self.iter}, T: {prev_T:.4e}->{self.T:.4e}, best_f={self.best_f:.6e}")

        return self.best_x, self.best_f

    def graficar(self):
        """
        Proyecta el recorrido de soluciones en 2D usando PCA y lo grafica.
        """
        historial_x = np.array(self.historial_x)
        if historial_x.shape[0] < 2:
            print("No hay suficientes datos en el historial para graficar.")
            return

        pca = PCA(n_components=2)
        puntos_2d = pca.fit_transform(historial_x)

        plt.figure(figsize=(8, 6))
        plt.plot(puntos_2d[:, 0], puntos_2d[:, 1], "o-", alpha=0.7, label="Recorrido")
        plt.scatter(puntos_2d[0, 0], puntos_2d[0, 1], c="red", s=100, marker="o", label="Inicio")
        plt.scatter(puntos_2d[-1, 0], puntos_2d[-1, 1], c="yellow", s=150, marker="*", label="Final")

        plt.title("Recorrido proyectado en 2D (PCA)")
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# === Ejemplos de funciones personalizadas ===
def perturb_uniform(x, T, T0):
    """Perturbación uniforme en un rango proporcional a T/T0."""
    escala = (T / T0)
    return x + np.random.uniform(-escala, escala, size=x.shape)

def cooling_linear(T, alpha):
    """Enfriamiento lineal: T decrece restando alpha."""
    return max(T - alpha, 1e-12)

def cooling_log(T, alpha):
    """Enfriamiento logarítmico: T decrece como T/(1+alpha)."""
    return max(T / (1 + alpha), 1e-12)