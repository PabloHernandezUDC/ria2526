# RIA 2025-2026 - Práctica 01: Aprendizaje por Refuerzo con Robobo

## Descripción del Proyecto

Esta práctica implementa un sistema de aprendizaje por refuerzo utilizando redes neuronales para que un robot Robobo sea capaz de:
1. Acercarse a un objeto (cilindro rojo)
2. Seguir el objeto independientemente de si está fijo o en movimiento
3. Funcionar desde cualquier posición inicial

## Tecnologías Utilizadas

- **Gymnasium**: Framework para entornos de aprendizaje por refuerzo
- **StableBaselines3**: Librería de algoritmos de aprendizaje por refuerzo
- **RoboboSim**: Simulador del robot Robobo
- **Robobo.py**: Librería para programar el robot Robobo
- **NumPy**: Operaciones matemáticas y arrays
- **Matplotlib/Seaborn**: Visualización de resultados

## Prerrequisitos

### Software necesario:
1. **RoboboSim** (Simulador Unity)
   - Descargar desde: https://github.com/mintforpeople/robobo-programming/wiki/Unity
   - Configurar el escenario "cylinder"

2. **Dependencias Python**:
   ```bash
   pip install gymnasium
   pip install stable-baselines3[extra]
   pip install numpy
   pip install matplotlib
   pip install seaborn
   pip install robobopy
   pip install robobosim
   ```

## Estructura del Proyecto

```
ria2526/
├── pr1.py                    # Código principal del entorno y entrenamiento
├── validate.py               # Script de validación (por crear)
├── results/                  # Directorio para guardar resultados
│   ├── models/              # Modelos entrenados
│   ├── plots/               # Gráficos generados
│   └── trajectories/        # Datos de trayectorias
└── README.md                # Este archivo
```

## Instrucciones de Ejecución

### Paso 1: Preparación del Simulador
1. Abrir RoboboSim
2. Cargar el escenario "cylinder"
3. Configurar física simplificada
4. Establecer velocidad de simulación x10
5. Asegurarse de que el simulador esté ejecutándose en localhost

### Paso 2: Entrenamiento del Modelo
```bash
# Ejecutar el entrenamiento
python pr1.py
```

El entrenamiento:
- Utiliza el algoritmo PPO de StableBaselines3
- Entrena durante un número específico de timesteps
- Guarda automáticamente el modelo entrenado
- Genera logs de TensorBoard para monitoreo

### Paso 3: Validación del Modelo
```bash
# Ejecutar la validación (una vez implementado)
python validate.py
```

La validación:
- Carga el modelo previamente entrenado
- Ejecuta episodios de prueba
- Registra las trayectorias para análisis
- Evalúa el rendimiento del robot

### Paso 4: Generación de Resultados
Los resultados se generan automáticamente e incluyen:
- Gráficos de métricas de entrenamiento (reward, loss, etc.)
- Representación 2D de trayectorias del robot
- Análisis de convergencia del algoritmo

## Arquitectura del Sistema

### Espacio de Observaciones
El robot observa:
- Posición del robot (x, z, orientación y)
- Posición del blob rojo detectado (x, y, tamaño)
- Distancia euclidiana al objetivo
- Información de sensores IR (opcional)

### Espacio de Acciones
Acciones discretas disponibles:
- 0: Avanzar (ambas ruedas adelante)
- 1: Girar izquierda (rueda derecha más rápida)
- 2: Girar derecha (rueda izquierda más rápida)
- 3: Retroceder (ambas ruedas atrás)

### Función de Recompensa
- **Recompensa positiva**: Por acercarse al objetivo
- **Penalización**: Por alejarse del objetivo
- **Bonificación**: Por orientación correcta hacia el objetivo
- **Penalización**: Por colisiones o comportamientos no deseados

### Algoritmo de Aprendizaje
- **PPO (Proximal Policy Optimization)**: Algoritmo de policy gradient
- **Red neuronal**: MultiInputPolicy para manejar observaciones complejas
- **Hiperparámetros**: Optimizados para el entorno específico

## Configuración del Entorno

### Parámetros del Robot
- Velocidad base: 20 unidades
- Duración de acción: 0.5 segundos (ajustable para velocidad x10)
- Ángulo de cámara: 105 grados (tilt)

### Parámetros de Entrenamiento
- Total timesteps: Configurable (recomendado: 50,000-100,000)
- Política: MultiInputPolicy
- Frecuencia de actualización: Cada episodio
- Criterio de parada: Distancia al objetivo < 50 unidades

## Monitoreo y Debugging

### Logs del Sistema
El sistema imprime información detallada:
```
Action: ↑ | Reward: 15.234 | Distance: 65.432 | Obs: {...}
```

### TensorBoard
```bash
# Para visualizar métricas en tiempo real
tensorboard --logdir ./tensorboard_logs/
```

### Debugging
- Verificar conexión con RoboboSim (localhost)
- Comprobar que el cilindro rojo sea visible
- Validar que las acciones muevan correctamente el robot

## Resultados Esperados

### Métricas de Éxito
- **Convergencia**: El reward promedio debe aumentar con el tiempo
- **Consistencia**: El robot debe aproximarse al objetivo >90% de las veces
- **Eficiencia**: Tiempo promedio de aproximación < 20 pasos

### Visualizaciones
- Gráfico de evolución del reward durante entrenamiento
- Mapa 2D de trayectorias recorridas por el robot
- Análisis de distribución de acciones tomadas

---

## CHECKLIST - Tareas Pendientes

### Implementación Básica
- [x] Estructura del entorno Gymnasium
- [x] Conexión con RoboboSim
- [x] Espacio de acciones básico
- [x] Función de recompensa inicial
- [ ] **Ampliar espacio de observaciones** (EN PROGRESO)
- [ ] Refinar función de recompensa
- [ ] Implementar condiciones de terminación completas

### Entrenamiento y Algoritmos
- [ ] Cambiar de A2C a PPO
- [ ] Configurar hiperparámetros óptimos
- [ ] Implementar guardado automático de modelos
- [ ] Configurar física simplificada y velocidad x10

### Validación y Testing
- [ ] Crear script de validación independiente
- [ ] Implementar carga de modelos entrenados
- [ ] Sistema de métricas de evaluación
- [ ] Pruebas con diferentes posiciones iniciales

### Visualización y Reportes
- [ ] Gráficos de métricas con matplotlib/seaborn
- [ ] Representación 2D de trayectorias
- [ ] Exportación de gráficos a PNG/JPEG
- [ ] Análisis de convergencia

### Documentación y Entrega
- [ ] Memoria técnica (4 páginas máximo)
- [ ] Documentación de dependencias
- [ ] Instrucciones de ejecución detalladas
- [ ] Empaquetado final en ZIP

### Optimizaciones Avanzadas
- [ ] Manejo de objetos en movimiento
- [ ] Espacios de observación continuos
- [ ] Función de recompensa con múltiples objetivos
- [ ] Análisis de robustez del sistema

---

**Fecha de entrega**: Viernes 10 de octubre, 23:59
**Entrega**: Archivo ZIP a través de Moodle