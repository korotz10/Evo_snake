"""
Script de prueba para verificar que todos los módulos funcionan correctamente.
Ejecutar sin interfaz Kivy para debugging rápido.

Uso: python test_standalone.py
"""

import numpy as np
import sys
from snake_game import SnakeGame, Direction
from neural_net import NeuralNetwork
from genetic_algo import SnakeGeneticAlgorithm


def test_snake_game():
    """Prueba el motor del juego."""
    print("=" * 50)
    print("PRUEBA 1: Motor del Juego (snake_game.py)")
    print("=" * 50)
    
    game = SnakeGame(grid_size=20)
    print(f"✓ Juego creado: grid {game.grid_size}x{game.grid_size}")
    
    # Simular algunos pasos
    for _ in range(10):
        sensors = game.get_sensors()
        random_direction = np.random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
        game.set_direction(random_direction)
        game.step()
    
    print(f"✓ Pasos simulados: {game.steps}")
    print(f"✓ Comida comida: {game.food_eaten}")
    print(f"✓ Fitness: {game.calculate_fitness()}")
    print(f"✓ Sensores shape: {game.get_sensors().shape}")
    print(f"✓ Grid state shape: {game.get_state().shape}")
    
    # Simular hasta el final del juego
    while not game.game_over:
        game.set_direction(np.random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]))
        game.step()
    
    print(f"✓ Juego completado")
    print(f"  - Pasos totales: {game.steps}")
    print(f"  - Comida total: {game.food_eaten}")
    print(f"  - Fitness final: {game.calculate_fitness()}")
    print()


def test_neural_network():
    """Prueba la red neuronal."""
    print("=" * 50)
    print("PRUEBA 2: Red Neuronal (neural_net.py)")
    print("=" * 50)
    
    net = NeuralNetwork(input_size=15, hidden_size=8, output_size=4)
    print(f"✓ Red neuronal creada")
    print(f"  - Total de parámetros: {net.get_total_params()}")
    
    # Prueba forward pass
    test_input = np.random.randn(15)
    output = net.forward(test_input)
    print(f"✓ Forward pass exitoso")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output probabilities: {output[0]}")
    
    # Prueba predict
    prediction = net.predict(test_input)
    print(f"✓ Predicción: clase {prediction[0]}")
    
    # Prueba serialización de pesos
    weights_flat = net.get_weights_flat()
    print(f"✓ Pesos serializados: {weights_flat.shape[0]} valores")
    
    # Prueba crear red desde pesos
    net2 = NeuralNetwork.create_from_weights(weights_flat)
    output2 = net2.forward(test_input)
    print(f"✓ Red creada desde pesos")
    print(f"  - Salida coincide: {np.allclose(output, output2)}")
    print()


def test_genetic_algorithm():
    """Prueba el algoritmo genético."""
    print("=" * 50)
    print("PRUEBA 3: Algoritmo Genético (genetic_algo.py)")
    print("=" * 50)
    
    ga = SnakeGeneticAlgorithm(
        population_size=10,  # Población pequeña para prueba rápida
        generations=3,       # Solo 3 generaciones
        mutation_prob=0.2,
        mutation_sigma=0.1,
        grid_size=15         # Grid más pequeño para prueba rápida
    )
    print(f"✓ Algoritmo genético creado")
    print(f"  - Tamaño de población: {ga.population_size}")
    print(f"  - Generaciones: {ga.generations}")
    print(f"  - Tamaño de cromosoma: {ga.chromosome_size}")
    
    # Historia de evolución
    history = []
    
    def callback(gen, best_fit, stats):
        history.append({'gen': gen, 'best_fit': best_fit, 'stats': stats})
    
    # Ejecutar evolución rápida
    print("\nEjecutando entrenamiento...")
    best_ind, training_history = ga.evolve(callback=callback)
    
    print(f"✓ Entrenamiento completado")
    print(f"  - Generaciones ejecutadas: {len(training_history)}")
    print(f"  - Mejor fitness inicial: {training_history[0]['best_fitness']:.0f}")
    print(f"  - Mejor fitness final: {training_history[-1]['best_fitness']:.0f}")
    
    # Obtener red entrenada
    best_net = ga.get_best_network()
    print(f"✓ Red neuronal entrenada obtenida")
    print(f"  - Tipo: {type(best_net)}")
    
    # Probar red entrenada
    game = SnakeGame(grid_size=15)
    while not game.game_over:
        sensors = game.get_sensors()
        output = best_net.predict(sensors)[0]
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        game.set_direction(directions[output])
        game.step()
    
    print(f"✓ Red entrenada jugó una partida")
    print(f"  - Comida comida: {game.food_eaten}")
    print(f"  - Pasos: {game.steps}")
    print(f"  - Fitness: {game.calculate_fitness()}")
    print()


def test_integration():
    """Prueba la integración completa."""
    print("=" * 50)
    print("PRUEBA 4: Integración Completa")
    print("=" * 50)
    
    print("Creando y entrenando IA...")
    
    ga = SnakeGeneticAlgorithm(
        population_size=20,
        generations=5,
        grid_size=15
    )
    
    best_ind, history = ga.evolve()
    best_net = ga.get_best_network()
    
    print(f"✓ Entrenamiento completado")
    print(f"  - Generaciones: {len(history)}")
    print(f"  - Fitness final: {history[-1]['best_fitness']:.0f}")
    
    # Jugar varias partidas con el mejor
    print("\nJugando 5 partidas con la mejor red...")
    fitness_scores = []
    
    for i in range(5):
        game = SnakeGame(grid_size=15)
        while not game.game_over:
            sensors = game.get_sensors()
            output = best_net.predict(sensors)[0]
            directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
            game.set_direction(directions[output])
            game.step()
        
        fitness = game.calculate_fitness()
        fitness_scores.append(fitness)
        print(f"  Partida {i+1}: Fitness = {fitness:.0f}, Comida = {game.food_eaten}")
    
    print(f"\n✓ Promedio de fitness en 5 partidas: {np.mean(fitness_scores):.0f}")
    print()


if __name__ == '__main__':
    try:
        print("\n" + "=" * 50)
        print("PRUEBAS DE EVOSNAKE")
        print("=" * 50 + "\n")
        
        test_snake_game()
        test_neural_network()
        test_genetic_algorithm()
        test_integration()
        
        print("=" * 50)
        print("✓ ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
