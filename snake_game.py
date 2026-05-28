"""
Motor del juego Snake.
Gestiona el grid, movimiento de la serpiente, colisiones y comida.
"""

import numpy as np
from enum import Enum
from collections import deque


class Direction(Enum):
    """Direcciones posibles del snake."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    """
    Juego Snake en un grid.
    
    Atributos:
        grid_size: tamaño del grid (grid_size x grid_size)
        snake: deque con posiciones del snake (head es el último)
        food: tupla (x, y) de la comida
        direction: dirección actual del snake
        next_direction: dirección para el siguiente movimiento
        steps: cantidad de pasos dados
        food_eaten: cantidad de comida comida
        game_over: si el juego terminó
    """
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reinicia el juego."""
        # Snake comienza en el centro, apuntando a la derecha
        center = self.grid_size // 2
        self.snake = deque([(center - 2, center), (center - 1, center), (center, center)])
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.steps = 0
        self.food_eaten = 0
        self.steps_since_last_food = 0  # Contador de pasos sin comer
        self.game_over = False
        self.max_steps_without_food = (self.grid_size * self.grid_size) // 2  # Max pasos sin comer
        
        # Generar comida en posición aleatoria
        self.food = self._generate_food()
    
    def _generate_food(self):
        """Genera comida en una posición aleatoria no ocupada por el snake."""
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.snake:
                return (x, y)
    
    def set_direction(self, direction):
        """
        Establece la dirección del siguiente movimiento.
        No permite cambiar 180 grados (dirección opuesta).
        """
        if direction == Direction.UP and self.direction != Direction.DOWN:
            self.next_direction = direction
        elif direction == Direction.DOWN and self.direction != Direction.UP:
            self.next_direction = direction
        elif direction == Direction.LEFT and self.direction != Direction.RIGHT:
            self.next_direction = direction
        elif direction == Direction.RIGHT and self.direction != Direction.LEFT:
            self.next_direction = direction
    
    def step(self):
        """
        Realiza un paso del juego.
        Retorna True si el juego continúa, False si terminó.
        """
        if self.game_over:
            return False
        
        self.steps += 1
        self.direction = self.next_direction
        
        # Calcular nueva posición de la cabeza
        head_x, head_y = self.snake[-1]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Verificar colisión con paredes
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.game_over = True
            return False
        
        # Verificar colisión con cuerpo
        if new_head in self.snake:
            self.game_over = True
            return False
        
        # Mover snake
        self.snake.append(new_head)
        self.steps_since_last_food += 1
        
        # Verificar si comió comida
        if new_head == self.food:
            self.food_eaten += 1
            self.steps_since_last_food = 0  # Resetear contador
            self.food = self._generate_food()
        else:
            # Si no comió, remover la cola para mantener el tamaño
            self.snake.popleft()
        
        # Verificar límite de pasos sin comer (penalizar inactividad)
        if self.steps_since_last_food >= self.max_steps_without_food:
            self.game_over = True
            return False
        
        return True
    
    def get_state(self):
        """
        Retorna el estado actual del juego como un grid.
        0: vacío
        1: comida
        2-N: cuerpo del snake (2 es la cola, N es la cabeza)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Marcar comida
        grid[self.food[1], self.food[0]] = 1
        
        # Marcar snake (cola a cabeza con valores 2 a 2+len(snake)-1)
        for i, (x, y) in enumerate(self.snake):
            grid[y, x] = 2 + i
        
        return grid
    
    def get_head(self):
        """Retorna la posición de la cabeza del snake."""
        return self.snake[-1]
    
    def get_food(self):
        """Retorna la posición de la comida."""
        return self.food
    
    def get_snake_body(self):
        """Retorna una copia de la lista del cuerpo del snake."""
        return list(self.snake)
    
    def calculate_fitness(self):
        """
        Calcula el fitness del individuo mejorado.
        Fitness = comida*500 - inactividad*0.1 + bonus_cercanía
        """
        # Componente principal: comida comida (muy importante)
        fitness = self.food_eaten * 500
        
        # Penalidad por pasos sin comer (desalienta quedarse quieto)
        fitness -= self.steps_since_last_food * 0.1
        
        # Bonus por cercanía a la comida (incluso si no la come)
        food_x, food_y = self.food
        head_x, head_y = self.get_head()
        manhattan_dist = abs(food_x - head_x) + abs(food_y - head_y)
        max_manhattan = 2 * self.grid_size
        proximity_bonus = (max_manhattan - manhattan_dist) / max_manhattan * 10
        fitness += proximity_bonus
        
        return fitness
    
    def _get_next_position(self, direction):
        """Retorna la posición siguiente en una dirección dada."""
        head_x, head_y = self.get_head()
        dx, dy = direction.value
        return (head_x + dx, head_y + dy)
    
    def _is_safe(self, position):
        """Verifica si una posición es segura (sin pared, sin cuerpo)."""
        x, y = position
        # Verificar pared
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        # Verificar cuerpo del snake (excepto la cola que se va a mover)
        if position in list(self.snake)[:-1]:
            return False
        return True
    
    @staticmethod
    def _turn_left(direction):
        """Retorna la dirección rotada 90 grados a la izquierda."""
        turn_map = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP
        }
        return turn_map[direction]
    
    @staticmethod
    def _turn_right(direction):
        """Retorna la dirección rotada 90 grados a la derecha."""
        turn_map = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP
        }
        return turn_map[direction]
    
    def get_sensors(self):
        """
        Retorna un vector de sensores para la red neuronal.
        
        Sensores:
        - Distancia a pared (UP, DOWN, LEFT, RIGHT): 4 valores
        - Dirección a comida (X, Y normalizados): 2 valores
        - Distancia Manhattan a comida: 1 valor
        - Dirección actual (one-hot encoding): 4 valores
        - Peligro inmediato (adelante, izq relativa, der relativa): 3 valores binarios
        Total: 15 entradas
        """
        head_x, head_y = self.get_head()
        food_x, food_y = self.food
        
        # Distancias normalizadas a paredes (0 a 1)
        dist_up = head_y / self.grid_size
        dist_down = (self.grid_size - 1 - head_y) / self.grid_size
        dist_left = head_x / self.grid_size
        dist_right = (self.grid_size - 1 - head_x) / self.grid_size
        
        # Dirección a la comida (normalizada)
        food_dir_x = (food_x - head_x) / self.grid_size
        food_dir_y = (food_y - head_y) / self.grid_size
        
        # Distancia Manhattan normalizada a comida
        manhattan_dist = (abs(food_x - head_x) + abs(food_y - head_y)) / (2 * self.grid_size)
        
        # Dirección actual (one-hot)
        dir_up = 1 if self.direction == Direction.UP else 0
        dir_down = 1 if self.direction == Direction.DOWN else 0
        dir_left = 1 if self.direction == Direction.LEFT else 0
        dir_right = 1 if self.direction == Direction.RIGHT else 0
        
        # Sensores de peligro inmediato (3 binarios)
        # Adelante: peligro en la dirección actual
        next_pos = self._get_next_position(self.direction)
        danger_ahead = 0 if self._is_safe(next_pos) else 1
        
        # Izquierda relativa (90 grados a la izquierda de la dirección actual)
        left_direction = self._turn_left(self.direction)
        left_pos = self._get_next_position(left_direction)
        danger_left = 0 if self._is_safe(left_pos) else 1
        
        # Derecha relativa (90 grados a la derecha de la dirección actual)
        right_direction = self._turn_right(self.direction)
        right_pos = self._get_next_position(right_direction)
        danger_right = 0 if self._is_safe(right_pos) else 1
        
        sensors = np.array([
            dist_up, dist_down, dist_left, dist_right,
            food_dir_x, food_dir_y, manhattan_dist,
            dir_up, dir_down, dir_left, dir_right,
            self.food_eaten / 10.0,  # Comida comida normalizada
            danger_ahead, danger_left, danger_right  # 3 sensores de peligro inmediato
        ], dtype=np.float32)
        
        return sensors
