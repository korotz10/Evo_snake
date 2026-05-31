# 🐍🧬 EvoSnake — Serpiente con IA Evolutiva

> **Neuroevolución aplicada al juego Snake**: una población de redes neuronales evoluciona mediante un algoritmo genético hasta aprender a jugar Snake de forma autónoma.

Proyecto final de **Computación Blanda** — Semestre VII.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Cómo Funciona la IA](#-cómo-funciona-la-ia)
- [Requisitos](#-requisitos)
- [Instalación y Uso](#-instalación-y-uso)
- [Configuración](#-configuración)

---

## 📖 Descripción

**EvoSnake** es un sistema de inteligencia artificial que aprende a jugar el clásico juego **Snake** sin ninguna programación explícita de reglas de juego. En lugar de usar técnicas tradicionales de aprendizaje supervisado, el proyecto emplea **neuroevolución**: un algoritmo genético optimiza directamente los pesos de redes neuronales a lo largo de múltiples generaciones.

La aplicación cuenta con una **interfaz gráfica moderna estilo Retro Arcade** desarrollada en Kivy, la cual permite entrenar la IA viendo las métricas en tiempo real y, posteriormente, ver a los mejores ejemplares ("campeones") jugar de forma autónoma.

---

## ✨ Características

| Categoría | Detalle |
|-----------|---------|
| 🧠 **Red Neuronal** | Feedforward con NumPy puro — Arquitectura ligera de 11 entradas, 8 neuronas ocultas y 4 salidas. |
| 🧬 **Algoritmo Genético** | Implementado con la librería **DEAP**. Selección por torneo, cruce y mutación gaussiana. |
| 🎮 **Motor de Juego** | Snake completo con sistema de **energía** (cada paso consume energía, comer la restaura) y sistema de sensores espaciales. |
| 📊 **Visualización UI** | Interfaz gráfica fluida en **Kivy** con renderizado gráfico optimizado (sin bloquear el hilo principal). Gráficas de fitness en tiempo real usando el canvas nativo. |
| 🖥️ **Modos de UI** | Pantalla de menú, entrenamiento (con slider de velocidad), juego (viendo jugar a los campeones) y resultados. |

---

## 📁 Arquitectura del Proyecto

```text
Evo_Snake/
├── main.py               # Punto de entrada — Interfaz gráfica Kivy y ScreenManager
├── snake_game.py         # Motor del juego Snake + sistema de 11 sensores + sistema de energía
├── neural_net.py         # Red neuronal feedforward (NumPy)
├── genetic_algo.py       # Algoritmo genético (basado en DEAP)
├── visualizer.py         # Renderizado de texturas para Kivy y gráficas nativas
├── config.py             # Configuración centralizada (diccionarios)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este archivo
```

---

## 🧠 Cómo Funciona la IA

### 1. Sistema de Sensores (11 entradas)

La serpiente cuenta con **11 sensores** clave para tomar decisiones en cada momento:

- **3 Sensores de peligro inmediato (Binarios):** Detectan si hay pared o cuerpo a la Izquierda relativa, al Frente o a la Derecha relativa.
- **4 Sensores de dirección actual (One-Hot):** Indican si la serpiente se mueve al Norte, Sur, Este u Oeste.
- **4 Sensores de posición de comida (Binarios):** Indican si la comida se encuentra al Norte, Sur, Este u Oeste respecto a la cabeza.

### 2. Arquitectura de la Red Neuronal

```text
Entrada (11) → Oculta (8) → Salida (4)
```
La red toma los 11 valores de los sensores, los procesa en una capa oculta de 8 neuronas y emite un resultado en una capa de salida de 4 neuronas (las 4 direcciones posibles). La dirección con el valor más alto es la acción elegida.

### 3. Algoritmo Genético (DEAP)

- **Población:** 100 - 150 individuos por generación.
- **Generaciones:** 50 - 100.
- **Selección:** Torneo (tamaño 3).
- **Cruce:** Probabilidad alta (70%-80%).
- **Mutación:** Probabilidad de 20%-30% con perturbación gaussiana.

### 4. Función de Fitness y Sistema de Energía

La serpiente tiene una energía inicial (ej: 300). Cada paso resta 1 punto de energía; si llega a 0, muere. Al comer, recupera 150 puntos.
El **Fitness** se calcula así:
```text
Fitness = (comidas * 1000) + pasos_sobrevividos + bonus_proximidad
```
El bonus de proximidad (distancia Manhattan a la comida) ayuda a guiar a las serpientes hacia el alimento en las primeras generaciones cuando aún no saben comer.

---

## 📋 Requisitos

- **Python** 3.8+
- Las dependencias se encuentran en `requirements.txt`:
  - `kivy`
  - `numpy`
  - `deap`
  - `Pillow`

---

## 🚀 Instalación y Uso

1. **Clonar e instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la Aplicación Gráfica:**
   ```bash
   python main.py
   ```
   
Esto abrirá la interfaz Kivy donde podrás:
1. Ir a **Entrenar IA** para iniciar el proceso evolutivo (puedes ajustar la velocidad con el slider).
2. Cuando el entrenamiento finalice, ir a **Ver Mejor Snake** para ver a la IA jugar autónomamente. Puedes usar el menú desplegable para ver cómo jugaba la generación 0 frente a la generación final.

---

## ⚙️ Configuración

Todos los hiperparámetros del sistema se pueden ajustar desde el archivo **`config.py`**. Está dividido en diccionarios temáticos:

- `GAME_CONFIG`: Tamaño de grilla y tamaños iniciales.
- `NEURAL_NETWORK_CONFIG`: Tamaño de capas de la red neuronal.
- `GA_CONFIG`: Tamaño de población, generaciones y probabilidades genéticas.
- `FITNESS_CONFIG`: Puntuaciones de recompensas y penalidades.
- `UI_CONFIG` & `VISUALIZATION_CONFIG`: Colores de la paleta Retro Arcade y métricas visuales.

---

**¡Disfruta entrenando tu propia IA para jugar Snake!** 🐍✨
