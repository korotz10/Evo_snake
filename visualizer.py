"""
Visualizadores para EvoSnake usando Kivy Canvas (sin matplotlib).
FitnessCanvasWidget: gráfica de fitness en tiempo real.
GameVisualizer: renderizado del juego a RGB.
"""

import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics import Line, Rectangle, Color
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from io import BytesIO
from PIL import Image as PILImage

from config import get_config


UI_THEME = get_config('ui')
VIS_THEME = get_config('visualization')
BACKGROUND = tuple(np.array(VIS_THEME['background_color'], dtype=np.float32) / 255.0)
GRID = tuple(np.array(VIS_THEME.get('grid_color', [40, 90, 140]), dtype=np.float32) / 255.0)
GREEN = tuple(np.array(VIS_THEME['snake_color_tail'], dtype=np.float32) / 255.0)
BLUE = tuple(np.array([40, 165, 255], dtype=np.float32) / 255.0)
YELLOW = tuple(np.array(VIS_THEME['food_color'], dtype=np.float32) / 255.0)
TEXT = tuple(UI_THEME['retro_text'])
RETRO_FONT = r'C:\Windows\Fonts\cour.ttf'


class FitnessCanvasWidget(Widget):
    """
    Widget que dibuja la gráfica de fitness usando Canvas nativo de Kivy.
    
    Muestra en tiempo real:
    - Línea verde: mejor fitness de cada generación
    - Línea azul: fitness promedio de cada generación
    - Fondo oscuro con grid de referencia
    - Etiquetas con generación actual y fitness máximo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.padding_x = 60  # Espacio para eje Y
        self.padding_y = 60  # Espacio para eje X
        self.margin = 20     # Margen general
        self.grid_lines = 5  # Número de líneas de grid
        
        # Bind canvas update
        self.bind(size=self.update_canvas)
    
    def update(self, history):
        """
        Actualiza la gráfica con el historial de fitness.
        
        Args:
            history: lista de dicts con 'best_fitness' y 'mean_fitness'
        """
        self.history = history
        self.update_canvas()
    
    def update_canvas(self, *args):
        """Redibuja el canvas con la gráfica actualizada."""
        self.canvas.clear()
        
        if not self.history or len(self.history) < 2:
            self._draw_empty_state()
            return
        
        # Calcular dimensiones
        plot_width = self.width - self.padding_x - self.margin
        plot_height = self.height - self.padding_y - self.margin
        plot_x = self.margin + self.padding_x
        plot_y = self.margin
        
        # Extraer datos
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        mean_fitness = [h['mean_fitness'] for h in self.history]
        
        # Encontrar rango de valores
        max_fitness = max(max(best_fitness), max(mean_fitness)) if best_fitness else 1
        min_fitness = min(min(best_fitness), min(mean_fitness)) if best_fitness else 0
        range_fitness = max_fitness - min_fitness if max_fitness > min_fitness else 1
        
        max_gen = max(generations) if generations else 1
        
        # Dibujar fondo
        with self.canvas:
            # Fondo oscuro
            Color(*BACKGROUND, 1)
            Rectangle(pos=(plot_x, plot_y), size=(plot_width, plot_height))
        
        # Dibujar grid
        self._draw_grid(plot_x, plot_y, plot_width, plot_height, 
                       max_gen, max_fitness, min_fitness)
        
        # Dibujar líneas de fitness
        self._draw_fitness_lines(plot_x, plot_y, plot_width, plot_height,
                                generations, best_fitness, mean_fitness,
                                max_gen, max_fitness, min_fitness)
        
        # Dibujar ejes
        self._draw_axes(plot_x, plot_y, plot_width, plot_height,
                       max_gen, max_fitness, min_fitness)
        
        # Añadir etiquetas
        self._draw_labels(plot_x, plot_y, plot_width, plot_height,
                         best_fitness[-1] if best_fitness else 0,
                         generations[-1] if generations else 0)
    
    def _draw_empty_state(self):
        """Dibuja un estado vacío cuando no hay datos."""
        with self.canvas:
            Color(*BACKGROUND, 1)
            Rectangle(pos=(0, 0), size=self.size)
    
    def _draw_grid(self, x, y, width, height, max_gen, max_fit, min_fit):
        """Dibuja el grid de referencia."""
        with self.canvas:
            Color(*GRID, 0.28)
            
            # Líneas verticales (generaciones)
            for i in range(self.grid_lines + 1):
                gen_x = x + (i / self.grid_lines) * width
                Line(points=[gen_x, y, gen_x, y + height], width=0.5)
            
            # Líneas horizontales (fitness)
            for i in range(self.grid_lines + 1):
                fit_y = y + (i / self.grid_lines) * height
                Line(points=[x, fit_y, x + width, fit_y], width=0.5)
    
    def _draw_fitness_lines(self, x, y, width, height, gens, best, mean, 
                           max_gen, max_fit, min_fit):
        """Dibuja las líneas de fitness (mejor y promedio)."""
        if not gens or len(gens) < 2:
            return
        
        range_fit = max_fit - min_fit if max_fit > min_fit else 1
        
        # Línea de mejor fitness (verde)
        best_points = []
        for i, (gen, fitness) in enumerate(zip(gens, best)):
            gen_x = x + (gen / max(max_gen, 1)) * width
            fit_y = y + ((fitness - min_fit) / range_fit) * height
            best_points.extend([gen_x, fit_y])
        
        if len(best_points) >= 4:
            with self.canvas:
                Color(*GREEN, 0.95)  # Verde arcade
                Line(points=best_points, width=2)
        
        # Línea de fitness promedio (azul)
        mean_points = []
        for i, (gen, fitness) in enumerate(zip(gens, mean)):
            gen_x = x + (gen / max(max_gen, 1)) * width
            fit_y = y + ((fitness - min_fit) / range_fit) * height
            mean_points.extend([gen_x, fit_y])
        
        if len(mean_points) >= 4:
            with self.canvas:
                Color(*BLUE, 0.85)  # Azul arcade
                Line(points=mean_points, width=1, dash_length=4, dash_offset=2)
    
    def _draw_axes(self, x, y, width, height, max_gen, max_fit, min_fit):
        """Dibuja los ejes X e Y."""
        with self.canvas:
            Color(1.0, 0.9, 0.2, 1)
            
            # Eje X (horizontal)
            Line(points=[x, y, x + width, y], width=1)
            
            # Eje Y (vertical)
            Line(points=[x, y, x, y + height], width=1)
    
    def _draw_labels(self, x, y, width, height, current_fitness, current_gen):
        """Dibuja etiquetas de información."""
        # Limpiar widgets previos (etiquetas)
        for widget in self.children[:]:
            if isinstance(widget, Label):
                self.remove_widget(widget)
        
        # Etiqueta de generación actual en esquina inferior derecha
        gen_label = Label(
            text=f'Gen: {int(current_gen)}',
            size_hint=(None, None),
            size=(80, 30),
            pos=(x + width - 100, y - 40),
            font_size='12sp',
            color=TEXT,
            font_name=RETRO_FONT
        )
        self.add_widget(gen_label)
        
        # Etiqueta de mejor fitness en esquina superior derecha
        fit_label = Label(
            text=f'Best: {int(current_fitness)}',
            size_hint=(None, None),
            size=(100, 30),
            pos=(x + width - 120, y + height + 10),
            font_size='12sp',
            color=YELLOW + (1,) if len(YELLOW) == 3 else YELLOW,
            font_name=RETRO_FONT
        )
        self.add_widget(fit_label)


class GameVisualizer:
    """
    Visualizador del juego Snake que convierte el estado del juego a RGB.
    """
    
    @staticmethod
    def get_game_frame(game, cell_size=10):
        """
        Genera un frame RGB del estado actual del juego con borde blanco de 2px.
        
        Args:
            game: objeto SnakeGame
            cell_size: tamaño de cada celda en píxeles
        
        Returns:
            np.array: frame RGB escalado con borde blanco
        """
        # Colores
        BACKGROUND = np.array(VIS_THEME['background_color'], dtype=np.uint8)    # Azul oscuro
        FOOD = np.array(VIS_THEME['food_color'], dtype=np.uint8)                # Amarillo neón
        SNAKE_BODY = np.array(VIS_THEME['snake_color_tail'], dtype=np.uint8)    # Verde arcade
        SNAKE_HEAD = np.array([80, 255, 140], dtype=np.uint8)                    # Verde brillante
        BORDER = np.array([40, 165, 255], dtype=np.uint8)                       # Azul neón
        
        # Crear grid base (grid_size x grid_size)
        grid_size = game.grid_size
        frame = np.full((grid_size, grid_size, 3), BACKGROUND, dtype=np.uint8)
        
        # Dibujar comida
        food_x, food_y = game.get_food()
        frame[food_y, food_x] = FOOD
        
        # Dibujar snake
        snake_body = game.get_snake_body()
        for i, (x, y) in enumerate(snake_body):
            if i == len(snake_body) - 1:  # Cabeza
                frame[y, x] = SNAKE_HEAD
            else:
                frame[y, x] = SNAKE_BODY
        
        # Escalar frame (grid_size * cell_size x grid_size * cell_size)
        frame_scaled = np.repeat(np.repeat(frame, cell_size, axis=0), 
                                 cell_size, axis=1)

        # Añadir una rejilla de píxeles muy sutil para reforzar el aspecto 8-bit
        frame_scaled[::cell_size, :, :] = np.clip(frame_scaled[::cell_size, :, :] + 12, 0, 255)
        frame_scaled[:, ::cell_size, :] = np.clip(frame_scaled[:, ::cell_size, :] + 12, 0, 255)
        
        # Añadir borde neón de 2px usando np.pad
        # Esto añade 2px arriba, 2px abajo, 2px izquierda, 2px derecha
        frame_with_border = np.pad(frame_scaled, 
                                  pad_width=((2, 2), (2, 2), (0, 0)),
                                  mode='constant',
                      constant_values=0)
        frame_with_border[:2, :, :] = np.array([40, 165, 255], dtype=np.uint8)
        frame_with_border[-2:, :, :] = np.array([40, 165, 255], dtype=np.uint8)
        frame_with_border[:, :2, :] = np.array([1, 255, 160], dtype=np.uint8)
        frame_with_border[:, -2:, :] = np.array([1, 255, 160], dtype=np.uint8)
        
        return frame_with_border
    
    @staticmethod
    def save_frame_as_kivy_image(frame):
        """
        Convierte un frame numpy a formato Kivy Image.
        
        Args:
            frame: np.array RGB
        
        Returns:
            str: ruta al archivo temporal PNG
        """
        # Convertir numpy array a PIL Image
        pil_img = PILImage.fromarray(frame.astype(np.uint8))
        
        # Guardar en BytesIO
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Retornar ruta temporal
        import tempfile
        import os
        
        # Crear archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(img_bytes.getvalue())
        temp_file.close()
        
        return temp_file.name

