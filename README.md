# EvoSnake 🐍🧬

Una aplicación móvil Android desarrollada en Python con Kivy que implementa el juego **Snake** controlado por una **IA que aprende sola** mediante una combinación de:
- **Red Neuronal Simple** (NumPy)
- **Algoritmo Genético** (DEAP)

La IA evoluciona los pesos de la red neuronal para mejorar su desempeño jugando Snake.

## 🎯 Características

✅ **IA Evolucionable**: Usa algoritmo genético para optimizar pesos de la red neuronal  
✅ **Red Neuronal Feedforward**: Arquitectura [12 inputs → 8 ocultas → 4 salidas]  
✅ **Interfaz Táctil**: Diseñada para pantallas pequeñas de Android  
✅ **Visualización en Tiempo Real**: Gráficas de fitness actualizadas durante entrenamiento  
✅ **Threading**: Entrenamiento en thread separado sin bloquear UI  
✅ **Compatible con Pydroid 3**: Funciona en dispositivos Android  

## 📁 Estructura del Proyecto

```
EvoSnake/
├── main.py              # App Kivy con navegación entre pantallas
├── snake_game.py        # Motor del juego
├── neural_net.py        # Red neuronal en NumPy
├── genetic_algo.py      # Algoritmo genético con DEAP
├── visualizer.py        # Gráficas de fitness y juego
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## 🧠 Cómo Funciona la IA

### 1. **Sensores del Snake** (12 inputs)
La red neuronal recibe como entrada:
- **Distancia a paredes** (4 valores): arriba, abajo, izquierda, derecha
- **Dirección a la comida** (2 valores): diferencia X, Y normalizada
- **Distancia Manhattan a comida** (1 valor): normalizada
- **Dirección actual** (4 valores): one-hot encoding (UP, DOWN, LEFT, RIGHT)
- **Comida comida** (1 valor): cantidad acumulada normalizada

### 2. **Arquitectura de la Red**
```
Entrada (12) → ReLU(8) → Softmax(4) → Salida
```
- **Capa oculta**: 8 neuronas con activación ReLU
- **Capa salida**: 4 neuronas con softmax (probabilities para UP/DOWN/LEFT/RIGHT)
- **Total de parámetros**: 12×8 + 8 + 8×4 + 4 = **152 parámetros**

### 3. **Algoritmo Genético**
- **Población**: 50 individuos
- **Generaciones**: 30 máximo
- **Cromosoma**: Vector de 152 pesos de la red neuronal
- **Selección**: Torneo de tamaño 3
- **Cruce**: Un punto sobre los vectores de pesos
- **Mutación**: Gaussiana con σ=0.1, probabilidad 20%

### 4. **Fitness**
```
Fitness = (Comida comida × 100) + Pasos sin morir
```
Favorece comer comida y sobrevivir el máximo tiempo posible.

## 📱 Pantallas de la App

### 1. **Menú Principal**
- Título "EvoSnake"
- Botón "Entrenar IA"
- Botón "Ver Mejor Snake"

### 2. **Pantalla de Entrenamiento**
- Barra de progreso (generación actual)
- Mejor fitness en tiempo real
- Gráfica matplotlib del fitness por generación (actualizada en vivo)
- Botón para iniciar/pausar

### 3. **Pantalla de Juego**
- Snake juega automáticamente en un grid 20×20
- Visualización del juego
- Métricas: comida comida, pasos, fitness
- Controles: Play/Pause, Reset, Volver

### 4. **Pantalla de Resultados**
- Gráficas resumidas de la evolución
- Estadísticas finales
- Botones para volver o jugar

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos

1. **Clonar/Descargar el proyecto**
```bash
cd EvoSnake
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Instalar el add-on de matplotlib para Kivy**
```bash
garden install matplotlib
```

## 🎮 Uso

### En PC (Linux/Mac/Windows)
```bash
python main.py
```

### En Pydroid 3 (Android)
1. Abrir Pydroid 3
2. Crear nuevo proyecto
3. Copiar todos los archivos `.py` al directorio del proyecto
4. Instalar pip packages: numpy, deap, kivy, matplotlib, Pillow
5. Ejecutar `main.py`

**Nota**: Pydroid 3 es más lento. El entrenamiento puede tomar más tiempo en Android.

## 📊 Estructura del Código

### `snake_game.py`
- Clase `SnakeGame`: Motor del juego
- Métodos clave:
  - `step()`: Mueve el snake un paso
  - `get_sensors()`: Retorna inputs para la red neuronal
  - `calculate_fitness()`: Calcula fitness del juego
  - `get_state()`: Retorna grid del juego

### `neural_net.py`
- Clase `NeuralNetwork`: Red feedforward simple
- Métodos clave:
  - `forward(inputs)`: Propaga inputs hacia adelante
  - `predict(inputs)`: Retorna clase predicha (dirección)
  - `get_weights_flat()`: Retorna pesos como vector 1D
  - `set_weights_flat(weights)`: Establece pesos desde vector 1D

### `genetic_algo.py`
- Clase `SnakeGeneticAlgorithm`: GA con DEAP
- Métodos clave:
  - `evolve(callback)`: Ejecuta el algoritmo genético
  - `evaluate(individual)`: Evalúa un individuo (plays a game)
  - `get_best_network()`: Retorna la mejor red entrenada

### `visualizer.py`
- Clase `FitnessVisualizer`: Gráficas de fitness
- Clase `GameVisualizer`: Visualización del juego
- Métodos clave:
  - `update_plot(history)`: Gráfica en tiempo real
  - `create_summary_plot(history)`: Gráfica de resumen
  - `get_game_frame(game)`: Obtiene frame RGB del juego

### `main.py`
- Clase `MenuScreen`: Pantalla de menú
- Clase `TrainingScreen`: Pantalla de entrenamiento
- Clase `GameScreen`: Pantalla de juego
- Clase `ResultsScreen`: Pantalla de resultados
- Clase `EvoSnakeApp`: Aplicación principal

## 🔧 Personalizaciones

Puedes modificar estos parámetros en `genetic_algo.py`:

```python
ga = SnakeGeneticAlgorithm(
    population_size=50,        # Tamaño de población
    generations=30,            # Número de generaciones
    mutation_prob=0.2,         # Probabilidad de mutación
    mutation_sigma=0.1,        # Desviación estándar de mutación
    grid_size=20               # Tamaño del grid (20x20)
)
```

O en `main.py` puedes cambiar la velocidad del juego:

```python
self.game_speed = 0.1  # segundos por step (reduce para jugar más rápido)
```

## 📈 Resultados Esperados

Después de 30 generaciones, la IA debería:
- **Generación 0**: Fitness ~50-200 (mueve aleatoriamente)
- **Generación 10**: Fitness ~300-500 (aprende a evitar paredes)
- **Generación 20**: Fitness ~600-1000 (aprende a buscar comida)
- **Generación 30**: Fitness ~1000-2000+ (juega razonablemente bien)

El fitness exacto depende de la aleatoriedad y el desempeño del GA.

## 🐛 Troubleshooting

### Error: "No module named 'deap'"
```bash
pip install deap
```

### Error: "No module named 'kivy'"
```bash
pip install kivy
```

### En Pydroid 3: matplotlib no se visualiza correctamente
- Pydroid 3 tiene limitaciones con matplotlib
- Considera usar versiones más simples de gráficas o ASCII art

### El juego es muy lento
- Aumenta `self.game_speed` en `GameScreen` (por ejemplo, a 0.05)
- O reduce `grid_size` a 15 o 10 para juegos más pequeños

## 📝 Notas Técnicas

- **Sin Backpropagation**: El GA optimiza los pesos directamente, no hay descenso de gradiente
- **Evaluación Costosa**: Cada individuo juega una partida completa, lo que es computacionalmente costoso
- **Threading**: El entrenamiento corre en un thread separado para no bloquear la UI
- **Estabilidad Numérica**: Se usa resta del máximo en softmax para evitar overflow

## 🎓 Conceptos Educativos

Este proyecto enseña:
- ✅ Redes neuronales feedforward
- ✅ Algoritmos genéticos
- ✅ Optimización sin gradientes
- ✅ Desarrollo de aplicaciones con Kivy
- ✅ Threading en Python
- ✅ Visualización de datos con matplotlib

## 📄 Licencia

Este proyecto es de código abierto y libre de usar con fines educativos.

## 👨‍💻 Autor

Proyecto desarrollado como trabajo final de Computación Blanda.

---

**¡Disfruta entrenando tu propia IA para jugar Snake!** 🐍✨
