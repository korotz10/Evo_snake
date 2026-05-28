# 📱 Guía de Instalación en Pydroid 3

Esta es una guía paso a paso para instalar y ejecutar **EvoSnake** en Pydroid 3 en Android.

## Requisitos

- ✅ Android 5.0 o superior
- ✅ Pydroid 3 instalado (disponible en Google Play)
- ✅ Al menos 200 MB de espacio libre

## Pasos de Instalación

### 1. Instalar Pydroid 3

1. Abre Google Play Store en tu Android
2. Busca "Pydroid 3"
3. Instala la aplicación (versión completa recomendada)
4. Abre Pydroid 3

### 2. Instalar Dependencias en Pydroid 3

En Pydroid 3, tienes dos opciones: usar pip o el gestor de paquetes.

#### Opción A: Usar pip (recomendado)

1. En Pydroid 3, toca el menú ☰ → "Pip"
2. En el cuadro de búsqueda, instala cada paquete:

```
numpy
kivy
deap
matplotlib
Pillow
```

Para cada paquete:
- Escribe el nombre
- Toca "Search"
- Toca "Install"
- Espera a que termine

#### Opción B: Usar Terminal de Pydroid

1. Toca el menú ☰ → "Terminal"
2. Ejecuta cada comando:

```bash
pip install numpy
pip install kivy
pip install deap
pip install matplotlib
pip install Pillow
```

### 3. Descargar Archivos de EvoSnake

1. En tu PC, copia todos los archivos `.py` de EvoSnake
2. Conecta tu Android por USB
3. O usa una app como "Syncthing" o "Dropbox" para sincronizar
4. Coloca los archivos en: `/sdcard/Pydroid3/Proyectos/EvoSnake/` o similar

Archivos necesarios:
```
main.py
snake_game.py
neural_net.py
genetic_algo.py
visualizer.py
config.py
```

### 4. Crear Proyecto en Pydroid 3

1. En Pydroid 3, toca "+" para crear nuevo proyecto
2. Nombre: "EvoSnake"
3. Copia el contenido de `main.py` en el editor
4. O importa los archivos si Pydroid lo permite

### 5. Ejecutar EvoSnake

1. En Pydroid 3, abre el proyecto "EvoSnake"
2. Asegúrate de que todos los módulos estén en el mismo directorio
3. Toca el botón ▶ (Play) para ejecutar
4. La aplicación debería iniciar

## Notas Importantes para Pydroid 3

### ⚠️ Limitaciones Conocidas

1. **Matplotlib puede no visualizarse correctamente**
   - Las gráficas pueden no renderizar como se espera
   - Solución: Se puede descomentar `# import matplotlib.pyplot as plt` si causa problemas

2. **El entrenamiento es lento**
   - Pydroid 3 ejecuta Python en la JVM, es más lento que Python nativo
   - Reduce el tamaño de población o generaciones si es muy lento
   - En `main.py`, en `TrainingScreen.run_training()`, modifica:
     ```python
     population_size=20,  # Reduce de 50 a 20
     generations=15,      # Reduce de 30 a 15
     ```

3. **La visualización del juego puede ser lenta**
   - En `main.py`, en `GameScreen`, aumenta `game_speed`:
     ```python
     self.game_speed = 0.2  # Aumenta de 0.1 a 0.2 o 0.3
     ```

4. **Memoria limitada**
   - Pydroid 3 puede tener límites de memoria
   - Reduce `grid_size` a 15 en lugar de 20 si hay problemas
   - En `genetic_algo.py`, modifica:
     ```python
     grid_size=15  # Reduce de 20 a 15
     ```

### 📱 Ajustes Recomendados para Pydroid 3

En `config.py`, ajusta:

```python
PYDROID_CONFIG = {
    'is_pydroid': True,             # Cambiar a True
    'use_kivy_garden': True,        # Usar addons de Kivy
    'game_speed_pydroid': 0.2,      # Velocidad más lenta
}
```

En `genetic_algo.py`, para Pydroid:

```python
ga = SnakeGeneticAlgorithm(
    population_size=20,   # En lugar de 50
    generations=15,       # En lugar de 30
    grid_size=15          # En lugar de 20
)
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'kivy'"
- Solución: Instala nuevamente con `pip install kivy`

### Error: "ModuleNotFoundError: No module named 'deap'"
- Solución: Instala con `pip install deap`

### La app se congela al entrenar
- Solución: Reduce el tamaño de población y generaciones
- O aumenta `game_speed` en `GameScreen`

### Las gráficas no aparecen
- Solución temporal: Comenta el import de matplotlib si causa problemas
- Alternativa: Usa ASCII art o simple texto para mostrar estadísticas

### Error de permisos en archivos
- Solución: En Pydroid 3, usa `/sdcard/` en lugar de directorios del sistema

## Optimización para Pydroid 3

### Script optimizado (`pydroid_optimized.py`)

Si tienes problemas de rendimiento, puedes crear una versión simplificada:

```python
# Versión simplificada sin matplotlib
# Solo usa DEAP, NumPy, Kivy
# Reduce grid a 10x10
# Reduce población a 15
# Reduce generaciones a 10
```

## Alternativa: Usar Colab/Jupyter

Si Pydroid 3 es muy problemático:

1. Usa Google Colab para entrenar la IA
2. Guarda el mejor modelo
3. Copia el vector de pesos
4. Crea una versión light de EvoSnake solo para jugar

## Conclusión

Pydroid 3 funciona, pero es más lento. Para desarrollo y debugging, se recomienda usar Python en PC, y usar Pydroid 3 para demos finales con parámetros reducidos.

---

¡Disfruta jugando con tu IA en Android! 🐍📱
