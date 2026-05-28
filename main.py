"""
Aplicación Kivy para EvoSnake.
Interfaz gráfica con navegación entre pantallas y threading para entrenamiento.
"""

import threading
import time
import numpy as np
from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.popup import Popup

from snake_game import SnakeGame, Direction
from neural_net import NeuralNetwork
from genetic_algo import SnakeGeneticAlgorithm
from visualizer import FitnessVisualizer, GameVisualizer


class MenuScreen(Screen):
    """Pantalla de menú principal."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'menu'
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Título
        title = Label(
            text='[b]EvoSnake[/b]\n[size=14]IA que aprende a jugar Snake[/size]',
            markup=True,
            size_hint_y=0.3
        )
        layout.add_widget(title)
        
        # Espaciador
        layout.add_widget(Label())
        
        # Botones
        btn_layout = GridLayout(cols=1, spacing=15, size_hint_y=0.4)
        
        btn_train = Button(
            text='Entrenar IA',
            size_hint_y=None,
            height=70,
            font_size='18sp'
        )
        btn_train.bind(on_press=self.go_to_training)
        btn_layout.add_widget(btn_train)
        
        btn_play = Button(
            text='Ver Mejor Snake',
            size_hint_y=None,
            height=70,
            font_size='18sp'
        )
        btn_play.bind(on_press=self.go_to_game)
        btn_layout.add_widget(btn_play)
        
        layout.add_widget(btn_layout)
        layout.add_widget(Label())
        
        self.add_widget(layout)
    
    def go_to_training(self, instance):
        self.manager.current = 'training'
    
    def go_to_game(self, instance):
        # Si no hay modelo entrenado, mostrar popup
        app = App.get_running_app()
        if app.best_network is None:
            popup = Popup(
                title='Sin modelo',
                content=Label(text='Primero debes entrenar la IA'),
                size_hint=(0.8, 0.4)
            )
            popup.open()
        else:
            self.manager.current = 'game'


class TrainingScreen(Screen):
    """Pantalla de entrenamiento con progreso en tiempo real."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'training'
        self.training_thread = None
        self.is_training = False
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Información
        self.lbl_generation = Label(
            text='Generación: 0',
            size_hint_y=0.1,
            font_size='16sp',
            bold=True
        )
        layout.add_widget(self.lbl_generation)
        
        self.lbl_fitness = Label(
            text='Mejor fitness: 0',
            size_hint_y=0.1,
            font_size='14sp'
        )
        layout.add_widget(self.lbl_fitness)
        
        # Barra de progreso
        self.progress_bar = ProgressBar(
            max=50,
            value=0,
            size_hint_y=0.1
        )
        layout.add_widget(self.progress_bar)
        
        # Gráfica de fitness
        self.plot_container = BoxLayout(size_hint_y=0.6)
        layout.add_widget(self.plot_container)
        
        # Botones
        btn_layout = GridLayout(cols=2, spacing=10, size_hint_y=0.1)
        
        self.btn_start = Button(text='Iniciar Entrenamiento', font_size='14sp')
        self.btn_start.bind(on_press=self.start_training)
        btn_layout.add_widget(self.btn_start)
        
        btn_back = Button(text='Volver', font_size='14sp')
        btn_back.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_back)
        
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def start_training(self, instance):
        if not self.is_training:
            self.is_training = True
            self.btn_start.disabled = True
            self.btn_start.text = 'Entrenando...'
            
            # Iniciar entrenamiento en thread separado
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def run_training(self):
        """Ejecuta el algoritmo genético en un thread separado."""
        try:
            app = App.get_running_app()
            ga = SnakeGeneticAlgorithm(
                population_size=100,
                generations=50,
                mutation_prob=0.2,
                mutation_sigma=0.2,
                grid_size=20
            )
            
            # Función callback para actualizar UI
            def update_callback(gen, best_fit, stats):
                history = ga.get_history()
                Clock.schedule_once(
                    lambda dt, g=gen, bf=best_fit, h=history: self.update_progress(g, bf, h),
                    0
                )
            
            # Ejecutar evolución
            best_ind, history = ga.evolve(callback=update_callback)
            
            # Guardar mejor red
            app.best_network = ga.get_best_network()
            app.training_history = history
            
            # Mostrar notificación
            Clock.schedule_once(
                lambda dt: self.on_training_complete(),
                0
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error durante entrenamiento: {error_msg}")
            Clock.schedule_once(
                lambda dt, msg=error_msg: self.on_training_error(msg),
                0
            )
    
    def update_progress(self, generation, best_fitness, history):
        """Actualiza la UI con el progreso actual."""
        self.lbl_generation.text = f'Generación: {generation + 1}/50'
        self.lbl_fitness.text = f'Mejor fitness: {best_fitness:.0f}'
        self.progress_bar.value = generation + 1
        
        # Actualizar gráfica
        visualizer = FitnessVisualizer()
        canvas = visualizer.update_plot(history)
        
        self.plot_container.clear_widgets()
        self.plot_container.add_widget(canvas)
    
    def on_training_complete(self):
        """Se llama cuando el entrenamiento termina."""
        self.is_training = False
        self.btn_start.disabled = False
        self.btn_start.text = 'Entrenamiento Completado ✓'
    
    def on_training_error(self, error):
        """Se llama si hay error durante el entrenamiento."""
        self.is_training = False
        self.btn_start.disabled = False
        self.btn_start.text = f'Error: {error[:30]}'
    
    def go_to_menu(self, instance):
        self.manager.current = 'menu'


class GameScreen(Screen):
    """Pantalla donde el mejor snake juega automáticamente."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'game'
        self.game_thread = None
        self.is_running = False
        self.current_game = None
        self.game_speed = 0.1  # segundos por step
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Canvas del juego
        self.game_canvas = Image()
        layout.add_widget(self.game_canvas)
        
        # Información del juego
        info_layout = GridLayout(cols=3, spacing=10, size_hint_y=0.15)
        
        self.lbl_food = Label(text='Comida: 0', font_size='14sp')
        info_layout.add_widget(self.lbl_food)
        
        self.lbl_steps = Label(text='Pasos: 0', font_size='14sp')
        info_layout.add_widget(self.lbl_steps)
        
        self.lbl_fitness = Label(text='Fitness: 0', font_size='14sp')
        info_layout.add_widget(self.lbl_fitness)
        
        layout.add_widget(info_layout)
        
        # Botones
        btn_layout = GridLayout(cols=3, spacing=10, size_hint_y=0.1)
        
        self.btn_play = Button(text='Play', font_size='12sp')
        self.btn_play.bind(on_press=self.toggle_play)
        btn_layout.add_widget(self.btn_play)
        
        btn_reset = Button(text='Reset', font_size='12sp')
        btn_reset.bind(on_press=self.reset_game)
        btn_layout.add_widget(btn_reset)
        
        btn_back = Button(text='Volver', font_size='12sp')
        btn_back.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_back)
        
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def on_enter(self):
        """Se llama cuando la pantalla se hace visible."""
        self.reset_game(None)
        self.toggle_play(None)
    
    def reset_game(self, instance):
        """Reinicia el juego."""
        self.is_running = False
        self.current_game = SnakeGame(grid_size=20)
        self.update_display()
    
    def toggle_play(self, instance):
        """Inicia o pausa el juego."""
        if not self.is_running:
            self.is_running = True
            self.btn_play.text = 'Pause'
            self.game_thread = threading.Thread(target=self.run_game)
            self.game_thread.daemon = True
            self.game_thread.start()
        else:
            self.is_running = False
            self.btn_play.text = 'Play'
    
    def run_game(self):
        """Ejecuta el juego en un thread separado."""
        app = App.get_running_app()
        
        while self.is_running and not self.current_game.game_over:
            # Obtener sensores del juego
            sensors = self.current_game.get_sensors()
            
            # Red neuronal predice la dirección
            output = app.best_network.predict(sensors)[0]
            
            # Convertir índice a dirección
            directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
            self.current_game.set_direction(directions[output])
            
            # Step del juego
            self.current_game.step()
            
            # Actualizar display
            Clock.schedule_once(lambda dt: self.update_display(), 0)
            
            # Esperar según la velocidad
            time.sleep(self.game_speed)
        
        # Juego terminó
        self.is_running = False
        Clock.schedule_once(
            lambda dt: setattr(self.btn_play, 'text', 'Juego Terminado'),
            0
        )
    
    def update_display(self):
        """Actualiza la visualización del juego."""
        if self.current_game:
            # Actualizar información
            self.lbl_food.text = f'Comida: {self.current_game.food_eaten}'
            self.lbl_steps.text = f'Pasos: {self.current_game.steps}'
            self.lbl_fitness.text = f'Fitness: {self.current_game.calculate_fitness()}'
            
            # Obtener frame del juego (escalado para visualización)
            frame = GameVisualizer.get_game_frame(self.current_game)
            
            # Escalar para mejor visualización (20x20 -> 400x400)
            frame_scaled = np.repeat(np.repeat(frame, 10, axis=0), 10, axis=1)
            
            # Convertir a formato PNG para Kivy
            import tempfile
            from PIL import Image as PILImage
            
            pil_image = PILImage.fromarray(frame_scaled)
            
            # Usar tempfile para crear archivo temporal en el directorio correcto
            if not hasattr(self, '_temp_file'):
                self._temp_file = tempfile.NamedTemporaryFile(
                    suffix='.png', 
                    delete=False
                )
                self._temp_path = self._temp_file.name
                self._temp_file.close()
            
            # Guardar imagen en archivo temporal
            pil_image.save(self._temp_path)
            
            # Actualizar imagen en Kivy
            self.game_canvas.source = self._temp_path
            self.game_canvas.reload()
    
    def go_to_menu(self, instance):
        self.is_running = False
        self.manager.current = 'menu'


class ResultsScreen(Screen):
    """Pantalla de resultados finales."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'results'
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Título
        title = Label(
            text='[b]Resultados del Entrenamiento[/b]',
            markup=True,
            size_hint_y=0.1,
            font_size='18sp'
        )
        layout.add_widget(title)
        
        # Gráfica de resumen
        self.plot_container = BoxLayout(size_hint_y=0.7)
        layout.add_widget(self.plot_container)
        
        # Botones
        btn_layout = GridLayout(cols=2, spacing=10, size_hint_y=0.1)
        
        btn_menu = Button(text='Menú', font_size='14sp')
        btn_menu.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_menu)
        
        btn_play = Button(text='Jugar', font_size='14sp')
        btn_play.bind(on_press=self.go_to_game)
        btn_layout.add_widget(btn_play)
        
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def on_enter(self):
        """Se llama cuando la pantalla se hace visible."""
        app = App.get_running_app()
        if app.training_history:
            visualizer = FitnessVisualizer()
            canvas = visualizer.create_summary_plot(app.training_history)
            
            self.plot_container.clear_widgets()
            self.plot_container.add_widget(canvas)
    
    def go_to_menu(self, instance):
        self.manager.current = 'menu'
    
    def go_to_game(self, instance):
        self.manager.current = 'game'


class EvoSnakeApp(App):
    """Aplicación principal de EvoSnake."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_network = None
        self.training_history = None
    
    def build(self):
        """Construye la interfaz principal."""
        # Configurar pantalla
        self.title = 'EvoSnake - IA que aprende Snake'
        
        # Manager de pantallas
        sm = ScreenManager()
        
        # Agregar pantallas
        sm.add_widget(MenuScreen())
        sm.add_widget(TrainingScreen())
        sm.add_widget(GameScreen())
        sm.add_widget(ResultsScreen())
        
        # Pantalla inicial
        sm.current = 'menu'
        
        return sm


if __name__ == '__main__':
    EvoSnakeApp().run()
