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

La app detecta Android/Pydroid automáticamente y activa un perfil más liviano:

- `grid_size` más pequeño
- menos generaciones y población por defecto
- velocidad de juego más lenta

Si ya tienes una sesión guardada desde PC y quieres volver al perfil móvil, usa el botón **RESTABLECER POR DEFECTO** en la pantalla de entrenamiento.

### ⚠️ Limitaciones Conocidas

1. **Matplotlib puede no visualizarse correctamente**
   - Las gráficas pueden no renderizar como se espera
   - Solución: Se puede descomentar `# import matplotlib.pyplot as plt` si causa problemas

2. **El entrenamiento es lento**
   - Pydroid 3 ejecuta Python en Android y suele ser más lento que en PC
   - La app ya arranca con un perfil reducido en móvil
   - Si hace falta, baja todavía más la población y las generaciones desde la interfaz de entrenamiento

3. **La visualización del juego puede ser lenta**
   - En móvil el juego ya usa una velocidad más lenta por defecto
   - Si necesitas más fluidez, puedes subir la velocidad en el código o bajar el tamaño del grid

4. **Memoria limitada**
   - Pydroid 3 puede tener límites de memoria
   - El modo móvil ya usa `grid_size=15` por defecto
   - Si aún tienes problemas, reduce población y generaciones desde la interfaz

### 📱 Ajustes Recomendados para Pydroid 3

En `config.py`, ajusta solo si quieres forzarlo manualmente:

```python
PYDROID_CONFIG = {
   'is_pydroid': True,                 # Forzar perfil móvil
   'use_kivy_garden': True,
   'game_speed_pydroid': 0.2,
   'grid_size_pydroid': 15,
   'training_generations_pydroid': 15,
   'training_population_size_pydroid': 20,
}
```

En `main.py`, para Pydroid el algoritmo se lanza con valores livianos por defecto, así que normalmente no necesitas tocar `genetic_algo.py`. Si quieres probar una versión más agresiva o más lenta, ajusta desde la UI.

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
