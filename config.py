"""
Archivo de configuración centralizada para EvoSnake.
Todos los parámetros ajustables en un único lugar.
"""

# ==================== Configuración del Juego ====================
GAME_CONFIG = {
    'grid_size': 20,                # Tamaño del grid (20x20)
    'max_steps_multiplier': 2,      # max_steps = grid_size^2 * multiplier
    'initial_snake_size': 3,        # Tamaño inicial del snake
}

# ==================== Configuración de la Red Neuronal ====================
NEURAL_NETWORK_CONFIG = {
    'input_size': 11,               # Número de sensores de entrada (3 peligro, 4 dir, 4 comida)
    'hidden_size': 8,               # Neuronas en capa oculta
    'output_size': 4,               # Número de salidas (UP, DOWN, LEFT, RIGHT)
    'weight_init_scale': 0.1,       # Escala de inicialización de pesos
}

# ==================== Configuración del Algoritmo Genético ====================
GA_CONFIG = {
    'population_size': 100,         # Tamaño de la población (más diversidad)
    'generations': 100,             # Máximo de generaciones (más tiempo para aprender)
    'tournament_size': 3,           # Tamaño del torneo para selección
    'crossover_prob': 0.8,          # Probabilidad de cruce (80% - más cruces)
    'mutation_prob': 0.3,           # Probabilidad de mutación por individuo
    'mutation_sigma': 0.2,          # Desviación estándar gaussiana (mutaciones más grandes)
    'weight_clipping': (-10, 10),   # Rango para clipping de pesos después de mutación
}

# ==================== Configuración de Fitness ====================
FITNESS_CONFIG = {
    'food_multiplier': 500,         # Multiplicador para comida comida (muy alto para incentivarlo)
    'inactivity_penalty': 0.1,      # Penalidad por pasos sin comer
    'proximity_bonus': 10,          # Bonus por cercanía a la comida
}

# ==================== Configuración de la UI (Kivy) ====================
UI_CONFIG = {
    'game_speed': 0.1,              # Segundos entre cada step del juego
    'window_size': (800, 600),      # Tamaño de ventana (será adaptado a pantalla)
    'padding': 10,                  # Padding general en widgets
    'spacing': 10,                  # Spacing entre widgets
    'font_size_title': '18sp',      # Tamaño de fuente para títulos
    'font_size_button': '14sp',     # Tamaño de fuente para botones
    'font_size_label': '12sp',      # Tamaño de fuente para labels
    'retro_background': [0.03, 0.05, 0.10, 1],   # Azul muy oscuro
    'retro_panel': [0.05, 0.08, 0.16, 1],        # Azul arcade
    'retro_panel_alt': [0.08, 0.10, 0.05, 1],    # Verde muy oscuro
    'retro_border': [1.0, 0.85, 0.15, 1],        # Amarillo neón
    'retro_green': [0.15, 1.0, 0.35, 1],         # Verde neón
    'retro_blue': [0.20, 0.65, 1.0, 1],          # Azul neón
    'retro_yellow': [1.0, 0.92, 0.20, 1],        # Amarillo neón
    'retro_text': [0.90, 1.0, 0.72, 1],          # Verde pálido
}

# ==================== Configuración de Visualización ====================
VISUALIZATION_CONFIG = {
    'plot_update_interval': 1,      # Actualizar gráfica cada N generaciones
    'game_frame_scale': 10,         # Escalar cada celda a NxN píxeles
    'snake_color_tail': [0, 180, 80],     # Verde arcade para cola
    'snake_color_head': [80, 255, 140],   # Verde brillante para cabeza
    'food_color': [255, 220, 40],         # Amarillo arcade para comida
    'background_color': [10, 18, 40],     # Azul oscuro para fondo
    'grid_color': [40, 90, 140],          # Azul para referencia
}

# ==================== Configuración de Logging ====================
LOGGING_CONFIG = {
    'verbose': True,                # Mostrar logs detallados
    'save_history': True,           # Guardar historial de entrenamiento
    'history_file': 'training_history.json',
}

# ==================== Configuración para Pydroid 3 ====================
PYDROID_CONFIG = {
    'is_pydroid': False,            # Set to True if running on Pydroid 3
    'use_kivy_garden': True,        # Usar jardín de Kivy para matplotlib
    'game_speed_pydroid': 0.2,      # Velocidad más lenta en Pydroid
    'grid_size_pydroid': 15,        # Grid más pequeño para móviles
    'training_generations_pydroid': 15,
    'training_population_size_pydroid': 20,
}


def get_config(section):
    """
    Retorna la configuración de una sección.
    
    Args:
        section: nombre de la sección (ej: 'GAME_CONFIG', 'GA_CONFIG')
    
    Returns:
        dict con la configuración
    """
    configs = {
        'game': GAME_CONFIG,
        'neural_network': NEURAL_NETWORK_CONFIG,
        'ga': GA_CONFIG,
        'fitness': FITNESS_CONFIG,
        'ui': UI_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'pydroid': PYDROID_CONFIG,
    }
    return configs.get(section.lower(), {})


def print_all_config():
    """Imprime toda la configuración."""
    all_configs = {
        'GAME_CONFIG': GAME_CONFIG,
        'NEURAL_NETWORK_CONFIG': NEURAL_NETWORK_CONFIG,
        'GA_CONFIG': GA_CONFIG,
        'FITNESS_CONFIG': FITNESS_CONFIG,
        'UI_CONFIG': UI_CONFIG,
        'VISUALIZATION_CONFIG': VISUALIZATION_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG,
        'PYDROID_CONFIG': PYDROID_CONFIG,
    }
    
    for section, config in all_configs.items():
        print(f"\n{section}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
