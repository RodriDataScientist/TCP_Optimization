import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FiveRandCities:
    """
    Clase que genera 5 ciudades aleatorias en coordenadas (x,y),
    construye la matriz de distancias y define una función objetivo
    para el TSP (problema del viajante).
    """

    def __init__(self, n_cities: int = 5, seed: int = None):
        """
        Constructor.

        Parámetros:
            n_cities : int
                Número de ciudades a generar (por defecto 5).
            seed : int
                Semilla opcional para reproducibilidad.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_cities = n_cities
        # Coordenadas de ciudades en rango [0, 100] (puedes cambiar rango)
        self.coords = np.random.uniform(0, 100, size=(n_cities, 2))
        # Matriz de distancias euclidianas
        self.dist_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Construye la matriz de distancias euclidianas."""
        n = self.n_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(self.coords[i] - self.coords[j])
                dist[i, j] = dist[j, i] = d
        return dist

    def tour_length(self, tour: np.ndarray) -> float:
        """
        Calcula la longitud total de un tour cerrado.

        Parámetros:
            tour : np.ndarray
                Permutación de índices de ciudades (ej. [0,2,1,4,3]).

        Retorna:
            float: longitud total del ciclo.
        """
        tour = np.asarray(tour, dtype=int)
        total = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)  # siguiente ciudad (cerrando ciclo)
            total += self.dist_matrix[tour[i], tour[j]]
        return total
    
    @staticmethod
    def tsp_func(tour: np.ndarray, cities) -> float:
        """
        Parámetros:
            tour : np.ndarray
                Vector de números (posiblemente reales) que representan el tour.
            cities : FiveRandCities
                Objeto que contiene coords y dist_matrix.

        Retorna:
            float: longitud total del tour.
        """
        # Convertimos a índices válidos
        tour_int = np.round(tour).astype(int) % cities.n_cities
        return cities.tour_length(tour_int)

    def random_tour(self) -> np.ndarray:
        """Genera un tour inicial aleatorio."""
        return np.random.permutation(self.n_cities)
    
    @staticmethod
    def perturb_1opt(tour, T, T0):
        """
        1-opt: intercambia dos ciudades al azar en el tour.
        """
        tour = tour.astype(int).copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    @staticmethod
    def perturb_2opt(tour, T, T0):
        """
        2-opt: selecciona dos índices y revierte el segmento entre ellos.
        """
        tour = tour.astype(int).copy()
        i, j = sorted(np.random.choice(len(tour), 2, replace=False))
        tour[i:j+1] = tour[i:j+1][::-1]
        return tour

    def plot_tour(self, tour: np.ndarray, title: str = "Tour TSP", 
             show_length: bool = True, arrow_scale: float = 0.8):
        """
        Plotea el tour con flechas que muestran la dirección de la ruta.
        
        Parámetros:
            tour : np.ndarray
                Tour a visualizar (debe ser una permutación de índices)
            title : str
                Título del gráfico
            show_length : bool
                Si mostrar la longitud del tour en el título
            arrow_scale : float
                Escala para el tamaño de las flechas (0-1)
        """
        # Asegurar que el tour sea circular
        if tour[0] != tour[-1]:
            tour = np.append(tour, tour[0])
        
        tour_coords = self.coords[tour]
        length = self.tour_length(tour)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotear ciudades
        ax.scatter(self.coords[:, 0], self.coords[:, 1], 
                c='red', s=200, alpha=0.9, 
                edgecolors='black', linewidth=2, 
                label='Ciudades', zorder=5)
        
        # Etiquetar ciudades
        for i, (x, y) in enumerate(self.coords):
            ax.annotate(f'C{i}', (x, y), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="yellow", alpha=0.8),
                    zorder=6)
        
        # Plotear la ruta con flechas
        for i in range(len(tour) - 1):
            start = tour_coords[i]
            end = tour_coords[i + 1]
            
            # Dibujar línea
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                'b-', linewidth=2.5, alpha=0.7, zorder=3)
            
            # Dibujar flecha
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Normalizar para flechas de tamaño consistente
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx = dx / norm * 12 * arrow_scale
                dy = dy / norm * 12 * arrow_scale
            
            ax.arrow(start[0], start[1], dx, dy, 
                    head_width=2.5, head_length=3, 
                    fc='darkblue', ec='darkblue', 
                    alpha=0.8, zorder=4)
        
        # Configuración del gráfico
        if show_length:
            title += f" - Longitud: {length:.2f}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Coordenada X', fontsize=11)
        ax.set_ylabel('Coordenada Y', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        ax.axis('equal')
        
        # Ajustar límites con margen
        margin = 5
        ax.set_xlim(min(self.coords[:, 0]) - margin, max(self.coords[:, 0]) + margin)
        ax.set_ylim(min(self.coords[:, 1]) - margin, max(self.coords[:, 1]) + margin)
        
        plt.tight_layout()
        plt.show()
        
        return length

    def load_distance_matrix(self, csv_path: str):
        """
        Carga una matriz de distancias desde un archivo CSV con encabezados.
        """
        df = pd.read_csv(csv_path, index_col=0)   # Ignora primera columna (Point_0,...)
        matrix = df.values.astype(float)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("La matriz debe ser cuadrada (NxN).")

        self.dist_matrix = matrix
        self.n_cities = matrix.shape[0]
        self.coords = None  # no tenemos coordenadas