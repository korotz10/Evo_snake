"""
Algoritmo Genético con DEAP para evolucionar los pesos de la red neuronal.
Los cromosomas son vectores de pesos que se cruzan y mutan.
"""

import numpy as np
from deap import base, creator, tools, algorithms
from neural_net import NeuralNetwork
from snake_game import SnakeGame, Direction


class SnakeGeneticAlgorithm:
    """
    Evoluciona redes neuronales para jugar Snake usando algoritmo genético.
    
    Atributos:
        population_size: tamaño de la población (default 50)
        generations: máximo de generaciones (default 30)
        mutation_prob: probabilidad de mutación (default 0.2)
        mutation_sigma: desviación estándar de la mutación (default 0.1)
        grid_size: tamaño del grid del juego (default 20)
        max_steps_per_game: máximo de pasos por partida (default grid_size**2 * 2)
    """
    
    def __init__(self, 
                 population_size=100,
                 generations=50,
                 mutation_prob=0.2,
                 mutation_sigma=0.2,
                 grid_size=20):
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.grid_size = grid_size
        
        # Configuración de la red neuronal
        self.input_size = 15
        self.hidden_size = 8
        self.output_size = 4
        
        # Crear un template de red para obtener el tamaño de cromosoma
        template_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.chromosome_size = template_net.get_total_params()
        
        # Configurar DEAP
        self._setup_deap()
        
        # Historial de evolución
        self.history = []
        self.best_individual = None
        self.best_fitness = 0
    
    def _setup_deap(self):
        """Configura los tipos y operadores genéticos de DEAP."""
        # Limpiar definiciones previas si existen
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Crear clase de fitness (maximizar)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Crear clase Individual como una lista de valores flotantes
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Crear toolbox
        self.toolbox = base.Toolbox()
        
        # Atributo: números aleatorios entre -1 y 1
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        
        # Estructura individual: lista de números flotantes
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=self.chromosome_size
        )
        
        # Población
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Operador de evaluación
        self.toolbox.register("evaluate", self.evaluate)
        
        # Operadores genéticos
        self.toolbox.register("mate", self._crossover_onepoint)
        self.toolbox.register("mutate", self._mutate_gaussian)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate(self, individual):
        """
        Evalúa la aptitud de un individuo.
        Juega una partida completa y retorna el fitness.
        
        Args:
            individual: lista de pesos de la red neuronal
        
        Returns:
            (fitness,): tupla con el valor de fitness
        """
        # Crear red neuronal con los pesos del individuo
        net = NeuralNetwork.create_from_weights(
            np.array(individual),
            self.input_size,
            self.hidden_size,
            self.output_size
        )
        
        # Jugar una partida
        game = SnakeGame(grid_size=self.grid_size)
        
        while not game.game_over:
            # Obtener sensores
            sensors = game.get_sensors()
            
            # Red neuronal predice la dirección
            output = net.predict(sensors)[0]  # Índice de máxima salida
            
            # Convertir índice a dirección
            directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
            game.set_direction(directions[output])
            
            # Paso del juego
            game.step()
        
        # Calcular fitness
        fitness = game.calculate_fitness()
        
        return (fitness,)
    
    @staticmethod
    def _crossover_onepoint(ind1, ind2):
        """
        Cruce de un punto entre dos individuos.
        
        Args:
            ind1, ind2: individuos a cruzar
        
        Returns:
            (ind1, ind2): individuos después del cruce
        """
        point = np.random.randint(1, len(ind1))
        
        ind1[point:], ind2[point:] = ind2[point:].copy(), ind1[point:].copy()
        
        del ind1.fitness.values
        del ind2.fitness.values
        
        return ind1, ind2
    
    def _mutate_gaussian(self, individual):
        """
        Mutación gaussiana: suma un valor aleatorio de N(0, sigma) a cada peso.
        
        Args:
            individual: individuo a mutar
        
        Returns:
            (individual,): tupla con el individuo mutado
        """
        for i in range(len(individual)):
            if np.random.random() < self.mutation_prob:
                individual[i] += np.random.normal(0, self.mutation_sigma)
                # Mantener los valores dentro de un rango razonable
                individual[i] = np.clip(individual[i], -10, 10)
        
        del individual.fitness.values
        
        return (individual,)
    
    def evolve(self, callback=None):
        """
        Ejecuta el algoritmo genético.
        
        Args:
            callback: función para llamar después de cada generación
                     recibe (generation, best_fitness, population_stats)
        
        Returns:
            best_individual: mejor individuo encontrado
            history: historial de fitness por generación
        """
        # Crear población inicial
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluar población inicial
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        self.history = []
        
        for gen in range(self.generations):
            # Seleccionar los mejores
            offspring = self.toolbox.select(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Aplicar cruce
            for i in range(1, len(offspring), 2):
                if np.random.random() < 0.8:  # Probabilidad de cruce 80%
                    self.toolbox.mate(offspring[i-1], offspring[i])
            
            # Aplicar mutación
            for i in range(len(offspring)):
                if np.random.random() < 0.3:  # Probabilidad de mutación 30%
                    self.toolbox.mutate(offspring[i])
            
            # Evaluar individuos con fitness inválido
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Reemplazar población
            population[:] = offspring
            
            # Obtener estadísticas
            fits = [ind.fitness.values[0] for ind in population]
            best_fit = max(fits)
            mean_fit = np.mean(fits)
            
            # Actualizar mejor individual encontrado
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_individual = max(population, key=lambda x: x.fitness.values[0])
            
            self.history.append({
                'generation': gen,
                'best_fitness': best_fit,
                'mean_fitness': mean_fit,
                'min_fitness': min(fits),
                'max_fitness': max(fits)
            })
            
            # Llamar callback si existe
            if callback:
                callback(gen, best_fit, {
                    'mean': mean_fit,
                    'min': min(fits),
                    'max': max(fits)
                })
        
        return self.best_individual, self.history
    
    def get_best_network(self):
        """
        Retorna la mejor red neuronal encontrada.
        
        Returns:
            NeuralNetwork: red con los pesos del mejor individuo
        """
        if self.best_individual is None:
            raise ValueError("Primero debe ejecutar evolve()")
        
        return NeuralNetwork.create_from_weights(
            np.array(self.best_individual),
            self.input_size,
            self.hidden_size,
            self.output_size
        )
    
    def get_history(self):
        """Retorna el historial de evolución."""
        return self.history
