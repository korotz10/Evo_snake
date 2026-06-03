"""
Aplicación Kivy para EvoSnake.
Interfaz gráfica con navegación entre pantallas y threading para entrenamiento.
Nuevo: FitnessCanvasWidget (sin matplotlib), Slider de velocidad, Spinner de campeones.
"""

import threading
import time
import json
import numpy as np
from typing import cast
from pathlib import Path
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
from kivy.core.window import Window
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Rectangle, Line

from config import get_config
from snake_game import SnakeGame, Direction
from neural_net import NeuralNetwork
from genetic_algo import SnakeGeneticAlgorithm
from visualizer import FitnessCanvasWidget, GameVisualizer


UI_THEME = get_config('ui')
WINDOW_BG = tuple(UI_THEME['retro_background'])
PANEL_BG = tuple(UI_THEME['retro_panel'])
PANEL_BG_ALT = tuple(UI_THEME['retro_panel_alt'])
BORDER = tuple(UI_THEME['retro_border'])
GREEN = tuple(UI_THEME['retro_green'])
BLUE = tuple(UI_THEME['retro_blue'])
YELLOW = tuple(UI_THEME['retro_yellow'])
TEXT = tuple(UI_THEME['retro_text'])
RETRO_FONT = r'C:\Windows\Fonts\cour.ttf'
TRAINING_GENERATIONS = get_config('ga').get('generations', 100)
TRAINING_POPULATION = get_config('ga').get('population_size', 100)
SESSION_SETTINGS_FILE = Path(__file__).with_name('training_session_settings.json')


def _is_compact_layout():
    return min(Window.size) < 720


def _session_generation_label(total_generations):
    return f'MEJOR (GEN {total_generations})'


def _load_training_session_settings():
    defaults = {
        'generations': TRAINING_GENERATIONS,
        'population_size': TRAINING_POPULATION,
    }

    try:
        if SESSION_SETTINGS_FILE.exists():
            data = json.loads(SESSION_SETTINGS_FILE.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                defaults['generations'] = int(data.get('generations', defaults['generations']))
                defaults['population_size'] = int(data.get('population_size', defaults['population_size']))
    except Exception:
        pass

    defaults['generations'] = max(10, min(200, defaults['generations']))
    defaults['population_size'] = max(50, min(300, defaults['population_size']))
    return defaults


def _save_training_session_settings(generations, population_size):
    payload = {
        'generations': int(generations),
        'population_size': int(population_size),
    }

    try:
        SESSION_SETTINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def _retro_panel(widget, fill_color=PANEL_BG, border_color=BORDER, border_width=1.6):
    def redraw(*args):
        widget.canvas.before.clear()
        widget.canvas.after.clear()
        with widget.canvas.before:
            Color(*fill_color)
            Rectangle(pos=widget.pos, size=widget.size)
        with widget.canvas.after:
            Color(*border_color)
            Line(rectangle=(widget.x, widget.y, widget.width, widget.height), width=border_width)

    widget.bind(pos=redraw, size=redraw)
    redraw()


def _retro_scanlines(widget, spacing=4, alpha=0.06):
    def redraw(*args):
        widget.canvas.after.clear()
        with widget.canvas.after:
            Color(0, 0, 0, alpha)
            for y in range(0, int(widget.height), spacing):
                Line(points=[widget.x, widget.y + y, widget.x + widget.width, widget.y + y], width=1)

    widget.bind(pos=redraw, size=redraw)
    redraw()


def _retro_label(label, color=TEXT, bold=False):
    if hasattr(label, 'font_name'):
        label.font_name = RETRO_FONT
    label.color = color
    label.bold = bold
    return label


def _retro_button(button, fill_color=BLUE, text_color=WINDOW_BG):
    if hasattr(button, 'font_name'):
        button.font_name = RETRO_FONT
    button.background_normal = ''
    button.background_down = ''
    button.background_color = fill_color
    button.color = text_color
    button.bold = True
    return button


def _retro_input(widget, fill_color=PANEL_BG_ALT, text_color=YELLOW):
    if hasattr(widget, 'font_name'):
        widget.font_name = RETRO_FONT
    widget.background_normal = ''
    widget.background_down = ''
    widget.background_color = fill_color
    widget.color = text_color
    return widget


class MenuScreen(Screen):
    """Pantalla de menú principal."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'menu'
        _retro_panel(self, WINDOW_BG, BORDER, 2)
        
        compact = _is_compact_layout()
        layout = BoxLayout(orientation='vertical', padding=14 if compact else 20, spacing=14 if compact else 20)
        _retro_panel(layout, PANEL_BG, BORDER, 2)
        _retro_scanlines(layout, spacing=4 if compact else 5, alpha=0.05)
        
        # Título
        title = Label(
            text='[color=ffdd33][b]EVOSNAKE[/b][/color]\n[color=6be6ff][size=14]IA QUE APRENDE A JUGAR SNAKE[/size][/color]',
            markup=True,
            size_hint_y=0.28 if compact else 0.3,
            color=TEXT,
            font_name=RETRO_FONT
        )
        layout.add_widget(title)
        
        # Espaciador
        layout.add_widget(Label())
        
        # Botones
        btn_layout = GridLayout(cols=1, spacing=12 if compact else 15, size_hint_y=0.4)
        
        btn_train = Button(
            text='ENTRENAR IA',
            size_hint_y=None,
            height=62 if compact else 70,
            font_size='16sp' if compact else '18sp'
        )
        _retro_button(btn_train, GREEN)
        btn_train.bind(on_press=self.go_to_training)
        btn_layout.add_widget(btn_train)
        
        btn_play = Button(
            text='VER MEJOR SNAKE',
            size_hint_y=None,
            height=62 if compact else 70,
            font_size='16sp' if compact else '18sp'
        )
        _retro_button(btn_play, YELLOW)
        btn_play.bind(on_press=self.go_to_game)
        btn_layout.add_widget(btn_play)
        
        layout.add_widget(btn_layout)
        layout.add_widget(Label())
        
        self.add_widget(layout)
    
    def go_to_training(self, instance):
        self.manager.current = 'training'
    
    def go_to_game(self, instance):
        # Si no hay modelo entrenado, mostrar popup
        app = cast(EvoSnakeApp, App.get_running_app())
        if app.best_network is None:
            popup = Popup(
                title='SIN MODELO',
                content=_retro_label(Label(text='PRIMERO DEBES ENTRENAR LA IA', color=TEXT)),
                size_hint=(0.8, 0.4)
            )
            _retro_panel(popup.content, PANEL_BG_ALT, BORDER, 1.4)
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
        session_settings = _load_training_session_settings()
        self.total_generations = session_settings['generations']
        self.population_size = session_settings['population_size']
        
        _retro_panel(self, WINDOW_BG, BORDER, 2)
        compact = _is_compact_layout()

        scroll = ScrollView(do_scroll_x=False, bar_width=6)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))
        layout.height = max(Window.height * 1.18, 760)
        _retro_panel(layout, PANEL_BG, BORDER, 2)
        _retro_scanlines(layout, spacing=4, alpha=0.05)

        def sync_layout_height(*args):
            layout.height = max(Window.height * 1.18, 760)

        Window.bind(size=sync_layout_height)
        sync_layout_height()
        
        # Información
        self.lbl_generation = Label(
            text=f'GENERACIÓN: 0 / {self.total_generations}',
            size_hint_y=None,
            height=40 if compact else 44,
            font_size='15sp' if compact else '16sp',
            bold=True,
            color=GREEN,
            font_name=RETRO_FONT
        )
        layout.add_widget(self.lbl_generation)
        
        self.lbl_fitness = Label(
            text='MEJOR FITNESS: 0',
            size_hint_y=None,
            height=34 if compact else 38,
            font_size='13sp' if compact else '14sp',
            color=YELLOW,
            font_name=RETRO_FONT
        )
        layout.add_widget(self.lbl_fitness)
        
        # Barra de progreso
        self.progress_bar = ProgressBar(
            max=self.total_generations,
            value=0,
            size_hint_y=None,
            height=20
        )
        layout.add_widget(self.progress_bar)
        
        # Control de velocidad del entrenamiento
        speed_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=44 if compact else 48, spacing=10)
        _retro_panel(speed_layout, PANEL_BG_ALT, BORDER, 1.2)
        speed_layout.add_widget(_retro_label(Label(text='VELOCIDAD:', size_hint_x=0.15), BLUE, True))
        
        self.speed_slider = Slider(min=0, max=0.5, value=0.0, size_hint_x=0.65)
        self.speed_slider.bind(value=self.on_speed_changed)
        speed_layout.add_widget(self.speed_slider)
        
        self.lbl_speed = Label(text='0.0S', size_hint_x=0.2, font_size='11sp' if compact else '12sp', color=YELLOW)
        self.lbl_speed.font_name = RETRO_FONT
        speed_layout.add_widget(self.lbl_speed)
        
        layout.add_widget(speed_layout)

        # Configuración de la sesión
        session_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=132 if compact else 152, spacing=8 if compact else 10)
        _retro_panel(session_layout, PANEL_BG_ALT, BORDER, 1.2)

        self.lbl_generation_setting = Label(
            text=f'GENS: {self.total_generations}',
            size_hint_y=None,
            height=24,
            font_size='11sp' if compact else '12sp',
            color=GREEN,
            font_name=RETRO_FONT
        )
        session_layout.add_widget(self.lbl_generation_setting)

        self.generations_slider = Slider(min=10, max=200, step=10, value=self.total_generations)
        self.generations_slider.bind(value=self.on_generations_changed)
        session_layout.add_widget(self.generations_slider)

        self.lbl_population_setting = Label(
            text=f'POBLACIÓN: {self.population_size}',
            size_hint_y=None,
            height=24,
            font_size='11sp' if compact else '12sp',
            color=GREEN,
            font_name=RETRO_FONT
        )
        session_layout.add_widget(self.lbl_population_setting)

        self.population_slider = Slider(min=50, max=300, step=10, value=self.population_size)
        self.population_slider.bind(value=self.on_population_changed)
        session_layout.add_widget(self.population_slider)

        layout.add_widget(session_layout)
        
        # Gráfica de fitness (usando Canvas nativo)
        self.fitness_canvas = FitnessCanvasWidget(size_hint_y=None, height=max(360, int(Window.height * 0.48)))
        layout.add_widget(self.fitness_canvas)
        
        # Botones
        btn_layout = GridLayout(cols=2, spacing=8 if compact else 10, size_hint_y=None, height=52 if compact else 58)
        _retro_panel(btn_layout, PANEL_BG_ALT, BORDER, 1.4)
        
        self.btn_start = Button(text='INICIAR ENTRENAMIENTO', font_size='13sp' if compact else '14sp')
        _retro_button(self.btn_start, GREEN)
        self.btn_start.bind(on_press=self.start_training)
        btn_layout.add_widget(self.btn_start)
        
        btn_back = Button(text='VOLVER', font_size='13sp' if compact else '14sp')
        _retro_button(btn_back, YELLOW)
        btn_back.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_back)
        
        layout.add_widget(btn_layout)
        
        scroll.add_widget(layout)
        self.add_widget(scroll)
    
    def on_speed_changed(self, instance, value):
        """Actualiza la etiqueta de velocidad cuando cambia el slider."""
        self.training_speed_delay = value
        self.lbl_speed.text = f'{value:.2f}s'

    def on_generations_changed(self, instance, value):
        self.total_generations = int(value)
        self.lbl_generation_setting.text = f'GENS: {self.total_generations}'
        self.lbl_generation.text = f'GENERACIÓN: 0 / {self.total_generations}'
        self.progress_bar.max = self.total_generations
        _save_training_session_settings(self.total_generations, self.population_size)

    def on_population_changed(self, instance, value):
        self.population_size = int(value)
        self.lbl_population_setting.text = f'POBLACIÓN: {self.population_size}'
        _save_training_session_settings(self.total_generations, self.population_size)
    
    def start_training(self, instance):
        if not self.is_training:
            self.is_training = True
            self.total_generations = int(self.generations_slider.value)
            self.population_size = int(self.population_slider.value)
            _save_training_session_settings(self.total_generations, self.population_size)
            self.progress_bar.max = self.total_generations
            self.progress_bar.value = 0
            self.lbl_generation.text = f'GENERACIÓN: 0 / {self.total_generations}'
            self.btn_start.disabled = True
            self.btn_start.text = 'Entrenando...'
            
            # Iniciar entrenamiento en thread separado
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def run_training(self):
        """Ejecuta el algoritmo genético en un thread separado."""
        try:
            app = cast(EvoSnakeApp, App.get_running_app())
            ga = SnakeGeneticAlgorithm(
                population_size=self.population_size,
                generations=self.total_generations,
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
            app.training_generations = ga.generations
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
        self.lbl_generation.text = f'GENERACIÓN: {generation} / {self.total_generations}'
        self.lbl_fitness.text = f'MEJOR FITNESS: {best_fitness:.0f}'
        self.progress_bar.value = generation
        
        # Actualizar gráfica con el nuevo historial
        self.fitness_canvas.update(history)
    
    def on_training_complete(self):
        """Se llama cuando el entrenamiento termina."""
        self.is_training = False
        self.btn_start.disabled = False
        self.btn_start.text = 'ENTRENAMIENTO COMPLETADO'
    
    def on_training_error(self, error):
        """Se llama si hay error durante el entrenamiento."""
        self.is_training = False
        self.btn_start.disabled = False
        self.btn_start.text = f'ERROR: {error[:24]}'
    
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
        self._champion_labels = {}
        self.total_generations = TRAINING_GENERATIONS
        
        compact = _is_compact_layout()
        layout = BoxLayout(orientation='vertical', padding=8 if compact else 10, spacing=8 if compact else 10)
        _retro_panel(self, WINDOW_BG, BORDER, 2)
        _retro_panel(layout, PANEL_BG, BORDER, 2)
        _retro_scanlines(layout, spacing=4, alpha=0.05)
        
        # Selector de campeones (Spinner)
        spinner_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=48 if compact else 54, spacing=8 if compact else 10)
        _retro_panel(spinner_layout, PANEL_BG_ALT, BORDER, 1.2)
        spinner_layout.add_widget(_retro_label(Label(text='SELECCIONA CAMPEÓN:', size_hint_x=0.3, font_size='11sp' if compact else '12sp'), BLUE, True))
        
        self.champion_spinner = Spinner(
            text=_session_generation_label(self.total_generations),
            values=(_session_generation_label(self.total_generations),),
            size_hint_x=0.7
        )
        self.champion_spinner.font_name = RETRO_FONT
        _retro_button(self.champion_spinner, YELLOW)
        self.champion_spinner.bind(text=self.on_champion_selected)
        spinner_layout.add_widget(self.champion_spinner)
        
        layout.add_widget(spinner_layout)
        
        # Canvas del juego
        self.game_canvas = Image(size_hint_y=0.8 if compact else 0.76)
        self.game_canvas.allow_stretch = True
        self.game_canvas.keep_ratio = True
        layout.add_widget(self.game_canvas)
        
        # Información del juego
        info_layout = GridLayout(cols=2 if compact else 4, spacing=8 if compact else 10, size_hint_y=None, height=76 if compact else 36)
        
        self.lbl_food = Label(text='COMIDA: 0', font_size='11sp' if compact else '12sp', color=GREEN, font_name=RETRO_FONT)
        info_layout.add_widget(self.lbl_food)
        
        self.lbl_steps = Label(text='PASOS: 0', font_size='11sp' if compact else '12sp', color=BLUE, font_name=RETRO_FONT)
        info_layout.add_widget(self.lbl_steps)
        
        self.lbl_energy = Label(text='ENERGÍA: 200', font_size='11sp' if compact else '12sp', color=YELLOW, font_name=RETRO_FONT)
        info_layout.add_widget(self.lbl_energy)
        
        self.lbl_fitness = Label(text='FITNESS: 0', font_size='11sp' if compact else '12sp', color=TEXT, font_name=RETRO_FONT)
        info_layout.add_widget(self.lbl_fitness)
        
        layout.add_widget(info_layout)
        
        # Botones
        btn_layout = GridLayout(cols=3, spacing=8 if compact else 10, size_hint_y=None, height=50 if compact else 54)
        
        self.btn_play = Button(text='PLAY', font_size='11sp' if compact else '12sp')
        _retro_button(self.btn_play, GREEN)
        self.btn_play.bind(on_press=self.toggle_play)
        btn_layout.add_widget(self.btn_play)
        
        btn_reset = Button(text='RESET', font_size='11sp' if compact else '12sp')
        _retro_button(btn_reset, BLUE)
        btn_reset.bind(on_press=self.reset_game)
        btn_layout.add_widget(btn_reset)
        
        btn_back = Button(text='VOLVER', font_size='11sp' if compact else '12sp')
        _retro_button(btn_back, YELLOW)
        btn_back.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_back)
        
        layout.add_widget(btn_layout)
        
        self.add_widget(layout)
    
    def on_champion_selected(self, spinner, text):
        """Se llama cuando se selecciona un campeón del spinner."""
        self.selected_champion_generation = self._champion_labels.get(text)
        
        # Reiniciar el juego con el nuevo campeón
        if not self.is_running:
            self.reset_game(None)

    def _refresh_champion_spinner(self):
        app = cast(EvoSnakeApp, App.get_running_app())
        default_label = _session_generation_label(getattr(app, 'training_generations', self.total_generations))
        spinner_options = [default_label]
        label_map: dict[str, int | None] = {default_label: None}

        if app.champions:
            champion_generations = sorted(app.champions.keys())
            for gen in champion_generations:
                label = f'GEN {gen}'
                spinner_options.append(label)
                label_map[label] = gen

        self._champion_labels = label_map
        self.champion_spinner.values = tuple(spinner_options)

        if self.champion_spinner.text not in self._champion_labels:
            self.champion_spinner.text = default_label
        else:
            self.selected_champion_generation = self._champion_labels.get(self.champion_spinner.text)
    
    def on_enter(self):
        """Se llama cuando la pantalla se hace visible."""
        self._refresh_champion_spinner()
        
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
            self.btn_play.text = 'PAUSA'
            self.game_thread = threading.Thread(target=self.run_game)
            self.game_thread.daemon = True
            self.game_thread.start()
        else:
            self.is_running = False
            self.btn_play.text = 'PLAY'
    
    def run_game(self):
        """Ejecuta el juego en un thread separado."""
        app = cast(EvoSnakeApp, App.get_running_app())
        current_game = self.current_game
        
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
        
        if current_game is None:
            self.is_running = False
            return

        while self.is_running and not current_game.game_over:
            # Obtener sensores del juego
            sensors = current_game.get_sensors()
            
            # Red neuronal predice la dirección
            output = network.predict(sensors)[0]
            
            # Convertir índice a dirección
            directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
            current_game.set_direction(directions[output])
            
            # Step del juego
            current_game.step()
            
            # Actualizar display
            Clock.schedule_once(lambda dt: self.update_display(), 0)
            
            # Esperar según la velocidad
            time.sleep(self.game_speed)
        
        # Juego terminó
        self.is_running = False
        Clock.schedule_once(
            lambda dt: setattr(self.btn_play, 'text', 'JUEGO TERMINADO'),
            0
        )
    
    def update_display(self):
        """Actualiza la visualización del juego."""
        current_game = self.current_game
        if current_game is not None:
            # Actualizar información
            self.lbl_food.text = f'COMIDA: {current_game.food_eaten}'
            self.lbl_steps.text = f'PASOS: {current_game.steps}'
            self.lbl_energy.text = f'ENERGÍA: {current_game.energy}'
            self.lbl_fitness.text = f'FITNESS: {current_game.calculate_fitness():.0f}'
            
            # Obtener frame del juego (ya incluye borde en visualizer.py)
            cell_area = min(max(self.game_canvas.width, 1), max(self.game_canvas.height, 1))
            cell_size = max(16, min(24, int(cell_area / (current_game.grid_size + 2))))
            frame = GameVisualizer.get_game_frame(current_game, cell_size=cell_size)
            
            # Convertir a formato Kivy Image
            temp_path = GameVisualizer.save_frame_as_kivy_image(frame)
            core_image = CoreImage(temp_path)
            texture: Texture | None = core_image.texture
            if texture is not None:
                texture.mag_filter = 'nearest'
                texture.min_filter = 'nearest'
            
            # Actualizar imagen en Kivy
            self.game_canvas.texture = texture
    
    def go_to_menu(self, instance):
        self.is_running = False
        self.manager.current = 'menu'


class ResultsScreen(Screen):
    """Pantalla de resultados finales."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'results'
        
        compact = _is_compact_layout()
        scroll = ScrollView(do_scroll_x=False, bar_width=6)
        layout = BoxLayout(orientation='vertical', padding=14 if compact else 20, spacing=12 if compact else 15, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))
        layout.height = max(Window.height * 1.12, 720)
        _retro_panel(self, WINDOW_BG, BORDER, 2)
        _retro_panel(layout, PANEL_BG, BORDER, 2)
        _retro_scanlines(layout, spacing=4 if compact else 5, alpha=0.05)

        def sync_layout_height(*args):
            layout.height = max(Window.height * 1.12, 720)

        Window.bind(size=sync_layout_height)
        sync_layout_height()
        
        # Título
        title = Label(
            text='[color=ffdd33][b]RESULTADOS DEL ENTRENAMIENTO[/b][/color]',
            markup=True,
            size_hint_y=None,
            height=40 if compact else 44,
            font_size='16sp' if compact else '18sp',
            color=TEXT,
            font_name=RETRO_FONT
        )
        layout.add_widget(title)
        
        # Gráfica de resumen usando Canvas
        self.fitness_canvas = FitnessCanvasWidget(size_hint_y=None, height=max(420, int(Window.height * 0.64)))
        layout.add_widget(self.fitness_canvas)
        
        # Botones
        btn_layout = GridLayout(cols=2, spacing=8 if compact else 10, size_hint_y=None, height=50 if compact else 56)
        
        btn_menu = Button(text='MENÚ', font_size='13sp' if compact else '14sp')
        _retro_button(btn_menu, GREEN)
        btn_menu.bind(on_press=self.go_to_menu)
        btn_layout.add_widget(btn_menu)
        
        btn_play = Button(text='JUGAR', font_size='13sp' if compact else '14sp')
        _retro_button(btn_play, YELLOW)
        btn_play.bind(on_press=self.go_to_game)
        btn_layout.add_widget(btn_play)
        
        layout.add_widget(btn_layout)
        
        scroll.add_widget(layout)
        self.add_widget(scroll)
    
    def on_enter(self):
        """Se llama cuando la pantalla se hace visible."""
        app = cast(EvoSnakeApp, App.get_running_app())
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
        self.best_network: NeuralNetwork | None = None
        self.training_history: list[dict] | None = None
        self.training_generations: int = TRAINING_GENERATIONS
        self.champions: dict[int, object] = {}  # Dict con campeones por generación
    
    def build(self):
        """Construye la interfaz principal."""
        # Configurar pantalla
        self.title = 'EvoSnake - IA que aprende Snake'
        Window.clearcolor = WINDOW_BG
        
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
