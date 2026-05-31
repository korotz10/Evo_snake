"""
Motor del juego Snake con sistema de energía.
Gestiona grid, movimiento de serpiente, colisiones, comida y energía.
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
    Juego Snake en un grid con sistema de energía.
    
    Atributos:
        grid_size: tamaño del grid (grid_size x grid_size)
        snake: deque con posiciones del snake (head es el último)
        food: tupla (x, y) de la comida
        direction: dirección actual del snake
        next_direction: dirección para el siguiente movimiento
        steps: cantidad de pasos dados
        food_eaten: cantidad de comida comida
        energy: energía actual del snake
        game_over: si el juego terminó
    """
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reinicia el juego con sistema de energía."""
        # Snake comienza en el centro, apuntando a la derecha
        center = self.grid_size // 2
        self.snake = deque([(center - 2, center), (center - 1, center), (center, center)])
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.steps = 0
        self.food_eaten = 0
        self.energy = 300  # Energía inicial
        self.max_energy = 600  # Energía máxima
        self.game_over = False
        self.death_reason = None  # Razón de fin de juego
        
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
        Realiza un paso del juego con sistema de energía.
        Retorna True si el juego continúa, False si terminó.
        """
        if self.game_over:
            return False
        
        self.steps += 1
        self.energy -= 1  # Cada movimiento cuesta 1 energía
        self.direction = self.next_direction
        
        # Calcular nueva posición de la cabeza
        head_x, head_y = self.snake[-1]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Verificar colisión con paredes
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.game_over = True
            self.death_reason = "pared"
            return False
        
        # Verificar colisión con cuerpo
        if new_head in self.snake:
            self.game_over = True
            self.death_reason = "cuerpo"
            return False
        
        # Mover snake
        self.snake.append(new_head)
        
        # Verificar si comió comida
        if new_head == self.food:
            self.food_eaten += 1
            # Añadir 150 de energía, con tope de max_energy
            self.energy = min(self.energy + 150, self.max_energy)
            self.food = self._generate_food()
        else:
            # Si no comió, remover la cola para mantener el tamaño
            self.snake.popleft()
        
        # Verificar si se agotó la energía
        if self.energy <= 0:
            self.game_over = True
            self.death_reason = "energia"
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
        Calcula el fitness con bonus de cercanía a la comida.
        
        Fitness = (comida_comidas * 1000) + pasos_sobrevividos + bonus_cercanía
        
        Bonus de cercanía = max(0, 100 - distancia_manhattan * 5)
        Premia acercarse a la comida aunque no la coma.
        """
        # Base: comida comida * 1000 + pasos sobrevividos
        base_fitness = self.food_eaten * 1000 + self.steps
        
        # Bonus por cercanía a la comida
        head_x, head_y = self.get_head()
        food_x, food_y = self.food
        manhattan_dist = abs(food_x - head_x) + abs(food_y - head_y)
        proximity_bonus = max(0, 100 - manhattan_dist * 5)
        
        total_fitness = base_fitness + proximity_bonus
        
        return total_fitness
    
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
        Retorna un vector de 11 sensores para la red neuronal.
        
        Sensores (11 entradas):
        1-3: Peligro relativo a orientación actual (izquierda, frente, derecha) - binario
        4-7: Dirección actual en one-hot (N, S, E, O)
        8-11: Posición relativa de comida en binario (¿está al N?, ¿S?, ¿E?, ¿O?)
        
        Returns:
            np.array de 11 valores flotantes en rango [0, 1]
        """
        head_x, head_y = self.get_head()
        food_x, food_y = self.food
        
        # SENSORES 1-3: Peligro relativo a la orientación actual (binario)
        # Izquierda relativa (90 grados a la izquierda)
        left_direction = self._turn_left(self.direction)
        left_pos = self._get_next_position(left_direction)
        danger_left = 0.0 if self._is_safe(left_pos) else 1.0
        
        # Frente (dirección actual)
        front_pos = self._get_next_position(self.direction)
        danger_front = 0.0 if self._is_safe(front_pos) else 1.0
        
        # Derecha relativa (90 grados a la derecha)
        right_direction = self._turn_right(self.direction)
        right_pos = self._get_next_position(right_direction)
        danger_right = 0.0 if self._is_safe(right_pos) else 1.0
        
        # SENSORES 4-7: Dirección actual en one-hot
        dir_north = 1.0 if self.direction == Direction.UP else 0.0
        dir_south = 1.0 if self.direction == Direction.DOWN else 0.0
        dir_east = 1.0 if self.direction == Direction.RIGHT else 0.0
        dir_west = 1.0 if self.direction == Direction.LEFT else 0.0
        
        # SENSORES 8-11: Posición relativa de comida (binario)
        # ¿Está la comida al norte? (comida_y < cabeza_y)
        food_north = 1.0 if food_y < head_y else 0.0
        # ¿Está la comida al sur? (comida_y > cabeza_y)
        food_south = 1.0 if food_y > head_y else 0.0
        # ¿Está la comida al este? (comida_x > cabeza_x)
        food_east = 1.0 if food_x > head_x else 0.0
        # ¿Está la comida al oeste? (comida_x < cabeza_x)
        food_west = 1.0 if food_x < head_x else 0.0
        
        sensors = np.array([
            danger_left, danger_front, danger_right,      # 3 sensores de peligro relativo
            dir_north, dir_south, dir_east, dir_west,      # 4 dirección actual one-hot
            food_north, food_south, food_east, food_west   # 4 posición relativa de comida
        ], dtype=np.float32)
        
        return sensors
