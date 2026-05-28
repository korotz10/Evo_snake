"""
Visualizador de gráficas de fitness usando matplotlib embebido en Kivy.
Muestra en tiempo real la evolución del fitness durante el entrenamiento.
"""

import numpy as np
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI


class FitnessVisualizer:
    """
    Crea visualizaciones del progreso de fitness durante el entrenamiento.
    """
    
    def __init__(self):
        self.history = []
        self.canvas = None
        self.fig = None
        self.ax = None
    
    def update_plot(self, history):
        """
        Actualiza la gráfica con el historial de fitness.
        
        Args:
            history: lista de dicts con 'generation', 'best_fitness', 'mean_fitness'
        
        Returns:
            FigureCanvasKivyAgg: widget de Kivy con la gráfica
        """
        self.history = history
        
        # Crear figura si no existe
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
        else:
            self.ax.clear()
        
        # Extraer datos
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        mean_fitness = [h['mean_fitness'] for h in history]
        
        # Graficar
        self.ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Mejor fitness')
        self.ax.plot(generations, mean_fitness, 'b--', linewidth=1.5, label='Fitness promedio')
        self.ax.fill_between(generations, best_fitness, mean_fitness, alpha=0.2, color='green')
        
        # Etiquetas y formato
        self.ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        self.ax.set_title('Evolución del Fitness - EvoSnake', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper left', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Convertir a widget Kivy
        self.canvas = FigureCanvasKivyAgg(self.fig)
        return self.canvas
    
    def create_summary_plot(self, history):
        """
        Crea una gráfica de resumen con más detalles.
        
        Args:
            history: historial de fitness
        
        Returns:
            FigureCanvasKivyAgg: widget de Kivy con la gráfica
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extraer datos
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        mean_fitness = [h['mean_fitness'] for h in history]
        min_fitness = [h['min_fitness'] for h in history]
        max_fitness = [h['max_fitness'] for h in history]
        
        # Plot 1: Fitness por generación
        axes[0, 0].plot(generations, best_fitness, 'g-', linewidth=2, label='Mejor')
        axes[0, 0].plot(generations, mean_fitness, 'b--', linewidth=1.5, label='Promedio')
        axes[0, 0].set_title('Fitness por Generación', fontweight='bold')
        axes[0, 0].set_xlabel('Generación')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rango de fitness (min-max)
        axes[0, 1].fill_between(generations, min_fitness, max_fitness, alpha=0.3, color='purple')
        axes[0, 1].plot(generations, min_fitness, 'r-', linewidth=1, label='Mínimo')
        axes[0, 1].plot(generations, max_fitness, 'g-', linewidth=1, label='Máximo')
        axes[0, 1].set_title('Rango de Fitness', fontweight='bold')
        axes[0, 1].set_xlabel('Generación')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mejora progresiva
        improvement = np.array(best_fitness) - np.array(best_fitness[0])
        axes[1, 0].plot(generations, improvement, 'orange', linewidth=2)
        axes[1, 0].fill_between(generations, 0, improvement, alpha=0.2, color='orange')
        axes[1, 0].set_title('Mejora desde Generación 0', fontweight='bold')
        axes[1, 0].set_xlabel('Generación')
        axes[1, 0].set_ylabel('Mejora de Fitness')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Estadísticas finales
        axes[1, 1].axis('off')
        final_best = best_fitness[-1] if best_fitness else 0
        final_mean = mean_fitness[-1] if mean_fitness else 0
        initial_best = best_fitness[0] if best_fitness else 0
        
        stats_text = f"""
        ESTADÍSTICAS FINALES
        
        Generaciones: {len(history)}
        Mejor fitness inicial: {initial_best:.0f}
        Mejor fitness final: {final_best:.0f}
        Mejora total: {final_best - initial_best:.0f}
        
        Fitness promedio final: {final_mean:.0f}
        Máximo alcanzado: {max(best_fitness):.0f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return FigureCanvasKivyAgg(fig)
    
    @staticmethod
    def create_comparison_plot(histories_dict):
        """
        Crea una gráfica comparativa de múltiples corridas.
        
        Args:
            histories_dict: dict con nombre -> historial
        
        Returns:
            FigureCanvasKivyAgg: widget de Kivy con la gráfica
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(histories_dict)))
        
        for (name, history), color in zip(histories_dict.items(), colors):
            generations = [h['generation'] for h in history]
            best_fitness = [h['best_fitness'] for h in history]
            ax.plot(generations, best_fitness, linewidth=2, label=name, color=color)
        
        ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mejor Fitness', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Entrenamientos - EvoSnake', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return FigureCanvasKivyAgg(fig)


class GameVisualizer:
    """
    Visualiza el juego en tiempo real en Kivy.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_game_frame(game):
        """
        Obtiene un array RGB de la pantalla del juego.
        
        Args:
            game: SnakeGame instance
        
        Returns:
            array RGB de forma (height, width, 3)
        """
        grid = game.get_state()
        height, width = grid.shape
        
        # Crear imagen RGB
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color de fondo (negro)
        frame[:, :] = [0, 0, 0]
        
        # Color de comida (rojo)
        food_x, food_y = game.food
        frame[food_y, food_x] = [255, 0, 0]
        
        # Color de snake (verde), cabeza más brillante
        snake_body = game.get_snake_body()
        for i, (x, y) in enumerate(snake_body):
            # Gradiente de verde: cola más oscura, cabeza más brillante
            intensity = int(100 + (150 * i / len(snake_body)))
            frame[y, x] = [0, intensity, 0]
        
        return frame
