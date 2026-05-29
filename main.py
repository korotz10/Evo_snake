"""
Aplicación Kivy para EvoSnake.
Interfaz gráfica con navegación entre pantallas y threading para entrenamiento.
Nuevo: FitnessCanvasWidget (sin matplotlib), Slider de velocidad, Spinner de campeones.
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
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.uix.popup import Popup

from snake_game import SnakeGame, Direction
from neural_net import NeuralNetwork
from genetic_algo import SnakeGeneticAlgorithm
from visualizer import FitnessCanvasWidget, GameVisualizer


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
    """Pantalla de entrenamiento con progreso en tiempo real y control de velocidad."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'training'
        self.training_thread = None
        self.is_training = False
        self.training_speed_delay = 0.0  # Segundos de espera entre generaciones
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Información
        self.lbl_generation = Label(
            text='Generación: 0',
            size_hint_y=0.08,
            font_size='16sp',
            bold=True
        )
        layout.add_widget(self.lbl_generation)
        
        self.lbl_fitness = Label(
            text='Mejor fitness: 0',
            size_hint_y=0.08,
            font_size='14sp'
        )
        layout.add_widget(self.lbl_fitness)
        
        # Barra de progreso
        self.progress_bar = ProgressBar(
            max=50,
            value=0,
            size_hint_y=0.08
        )
        layout.add_widget(self.progress_bar)
        
        # Control de velocidad del entrenamiento
        speed_layout = BoxLayout(orientation='horizontal', size_hint_y=0.08, spacing=10)
        speed_layout.add_widget(Label(text='Velocidad:', size_hint_x=0.15))
        
        self.speed_slider = Slider(min=0, max=0.5, value=0.0, size_hint_x=0.65)
        self.speed_slider.bind(value=self.on_speed_changed)
        speed_layout.add_widget(self.speed_slider)
        
        self.lbl_speed = Label(text='0.0s', size_hint_x=0.2, font_size='12sp')
        speed_layout.add_widget(self.lbl_speed)
        
        layout.add_widget(speed_layout)
        
        # Gráfica de fitness (usando Canvas nativo)
        self.fitness_canvas = FitnessCanvasWidget(size_hint_y=0.55)
        layout.add_widget(self.fitness_canvas)
        
        # Botones
        btn_layout = GridLayout(cols=2, spacing=10, size_hint_y=0.08)
        
        self.btn_start = Button(text='Iniciar Entrenamiento', font_size='14sp')
        self.btn_start.bind(on_press=self.start_training)
        btn_layout.add_widget(self.btn_start)
        
        btn_back = Button(text='Volver', font_size='14sp')
        btn_back.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_back)
        
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def on_speed_changed(self, instance, value):
        """Actualiza la etiqueta de velocidad cuando cambia el slider."""
        self.training_speed_delay = value
        self.lbl_speed.text = f'{value:.2f}s'
    
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
                mutation_sigma=0.1,
                crossover_prob=0.7,
                tournament_size=3,
                grid_size=20
            )
            
            # Función callback para actualizar UI
            def update_callback(gen, best_fit, stats):
                # Esperar según la velocidad configurada en el slider
                time.sleep(self.training_speed_delay)
                
                history = ga.get_history()
                Clock.schedule_once(
                    lambda dt, g=gen, bf=best_fit, h=history: self.update_progress(g, bf, h),
                    0
                )
            
            # Ejecutar evolución
            best_ind, history = ga.evolve(callback=update_callback)
            
            # Guardar mejor red y campeones
            app.best_network = ga.get_best_network()
            app.training_history = history
            app.champions = ga.champions  # Guardar dict de campeones
            
            # Mostrar notificación
            Clock.schedule_once(
                lambda dt: self.on_training_complete(),
                0
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"Error durante entrenamiento: {error_msg}")
            import traceback
            traceback.print_exc()
            Clock.schedule_once(
                lambda dt, msg=error_msg: self.on_training_error(msg),
                0
            )
    
    def update_progress(self, generation, best_fitness, history):
        """Actualiza la UI con el progreso actual."""
        self.lbl_generation.text = f'Generación: {generation + 1}/50'
        self.lbl_fitness.text = f'Mejor fitness: {best_fitness:.0f}'
        self.progress_bar.value = generation + 1
        
        # Actualizar gráfica con el nuevo historial
        self.fitness_canvas.update(history)
    
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
    """Pantalla donde el mejor snake (o un campeón) juega automáticamente."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'game'
        self.game_thread = None
        self.is_running = False
        self.current_game = None
        self.game_speed = 0.1  # segundos por step
        self.selected_champion_generation = None  # Generación del campeón seleccionado
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Selector de campeones (Spinner)
        spinner_layout = BoxLayout(orientation='horizontal', size_hint_y=0.08, spacing=10)
        spinner_layout.add_widget(Label(text='Selecciona campeón:', size_hint_x=0.3))
        
        self.champion_spinner = Spinner(
            text='Mejor (Gen 50)',
            values=('Mejor (Gen 50)', 'Gen 0', 'Gen 10', 'Gen 20', 'Gen 30', 'Gen 40'),
            size_hint_x=0.7
        )
        self.champion_spinner.bind(text=self.on_champion_selected)
        spinner_layout.add_widget(self.champion_spinner)
        
        layout.add_widget(spinner_layout)
        
        # Canvas del juego
        self.game_canvas = Image(size_hint_y=0.75)
        layout.add_widget(self.game_canvas)
        
        # Información del juego
        info_layout = GridLayout(cols=4, spacing=10, size_hint_y=0.08)
        
        self.lbl_food = Label(text='Comida: 0', font_size='12sp')
        info_layout.add_widget(self.lbl_food)
        
        self.lbl_steps = Label(text='Pasos: 0', font_size='12sp')
        info_layout.add_widget(self.lbl_steps)
        
        self.lbl_energy = Label(text='Energía: 200', font_size='12sp')
        info_layout.add_widget(self.lbl_energy)
        
        self.lbl_fitness = Label(text='Fitness: 0', font_size='12sp')
        info_layout.add_widget(self.lbl_fitness)
        
        layout.add_widget(info_layout)
        
        # Botones
        btn_layout = GridLayout(cols=3, spacing=10, size_hint_y=0.08)
        
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
    
    def on_champion_selected(self, spinner, text):
        """Se llama cuando se selecciona un campeón del spinner."""
        # Mapear texto del spinner a generación
        generation_map = {
            'Mejor (Gen 50)': None,  # Mejor del entrenamiento
            'Gen 0': 0,
            'Gen 10': 10,
            'Gen 20': 20,
            'Gen 30': 30,
            'Gen 40': 40
        }
        self.selected_champion_generation = generation_map.get(text)
        
        # Reiniciar el juego con el nuevo campeón
        if not self.is_running:
            self.reset_game(None)
    
    def on_enter(self):
        """Se llama cuando la pantalla se hace visible."""
        # Actualizar opciones del spinner si hay champions disponibles
        app = App.get_running_app()
        if app.champions:
            champion_generations = sorted(app.champions.keys())
            spinner_options = ['Mejor (Gen 50)']
            for gen in champion_generations:
                if gen > 0:  # Excluir generación 0 si está
                    spinner_options.append(f'Gen {gen}')
            
            # Agregar opciones del spinner
            if len(spinner_options) > 1:
                self.champion_spinner.values = tuple(spinner_options)
        
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
        
        # Obtener la red neuronal a usar (campeón o mejor)
        if self.selected_champion_generation is not None and app.champions:
            network = app.champions.get(self.selected_champion_generation)
            if network is not None:
                network = NeuralNetwork.create_from_weights(
                    np.array(network),
                    input_size=11,
                    hidden_size=8,
                    output_size=4
                )
            else:
                network = app.best_network
        else:
            network = app.best_network
        
        if network is None:
            self.is_running = False
            return
        
        while self.is_running and not self.current_game.game_over:
            # Obtener sensores del juego
            sensors = self.current_game.get_sensors()
            
            # Red neuronal predice la dirección
            output = network.predict(sensors)[0]
            
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
            self.lbl_energy.text = f'Energía: {self.current_game.energy}'
            self.lbl_fitness.text = f'Fitness: {self.current_game.calculate_fitness():.0f}'
            
            # Obtener frame del juego (ya incluye borde en visualizer.py)
            frame = GameVisualizer.get_game_frame(self.current_game, cell_size=10)
            
            # Convertir a formato Kivy Image
            temp_path = GameVisualizer.save_frame_as_kivy_image(frame)
            
            # Actualizar imagen en Kivy
            self.game_canvas.source = temp_path
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
        
        # Gráfica de resumen usando Canvas
        self.fitness_canvas = FitnessCanvasWidget(size_hint_y=0.7)
        layout.add_widget(self.fitness_canvas)
        
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
            self.fitness_canvas.update(app.training_history)
    
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
        self.champions = {}  # Dict con campeones por generación
    
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
