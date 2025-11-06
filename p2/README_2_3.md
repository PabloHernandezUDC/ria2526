# Práctica 2.3 - Navegación con Obstáculo (Estrategia Híbrida)

## Objetivo
El robot Robobo debe aprender a acercarse al cilindro rojo en el escenario "Avoid the block" con el sólido justo en medio, partiendo de la posición inicial predeterminada.

## Estrategia Implementada

### Función de Recompensa Híbrida

Se ha implementado una función de recompensa híbrida (`get_hybrid_reward`) que combina dos estrategias:

#### 1. **Evitación de Obstáculos (cuando `ir_front >= 2`)**
Cuando el robot detecta un obstáculo cercano mediante el sensor IR frontal:
- **Penalización fuerte** (-2.0) por estar cerca del obstáculo
- **Bonus por girar** (+0.5) para fomentar maniobras de evitación
- **Recompensa por progreso** basada en reducción de distancia al objetivo

#### 2. **Navegación Basada en Distancia (cuando `sector == 5` y no hay obstáculo)**
Cuando el objetivo no es visible pero no hay obstáculo:
- **Recompensa por distancia**: inversamente proporcional a la distancia al objetivo
- **Recompensa por ángulo**: penaliza no estar orientado hacia el objetivo
- **Recompensa por progreso**: bonus por acercarse al objetivo

#### 3. **Navegación Óptima (objetivo visible y sin obstáculo)**
Cuando el objetivo es visible y no hay obstáculo:
- **Recompensa combinada** de distancia y ángulo
- **Bonus por visibilidad** (+1.0) por mantener el objetivo a la vista
- **Bonus por centrado** (+0.5) si el objetivo está en sectores centrales (2-3)
- **Recompensa por progreso** continua

### Parámetros de Entrenamiento

- **Población**: 10 genomas
- **Generaciones**: 50
- **Episodios por genoma**: 3
- **Pasos máximos**: 100
- **Alpha (peso distancia)**: 0.5
- **Entradas red neuronal**: 10 (6 sectores visuales + 4 sectores IR)
- **Salidas red neuronal**: 1 (mapeada a 3 acciones discretas)

### Ventajas de la Estrategia Híbrida

1. **Evitación reactiva**: El robot aprende a girar cuando detecta obstáculos
2. **Navegación informada**: Usa distancia euclidiana cuando pierde el objetivo de vista
3. **Optimización final**: Maximiza velocidad de acercamiento cuando tiene visión clara
4. **Robustez**: Combina sensores visuales e IR para mejor percepción del entorno

### Archivos Modificados

- `robobo_utils/helpers.py`: Nueva función `get_hybrid_reward()`
- `robobo_utils/environment.py`: Integración de recompensa híbrida y tracking de distancia previa
- `robobo_utils/__init__.py`: Exportación de nueva función
- `p2_3_train.py`: Script de entrenamiento adaptado
- `p2_3_validate.py`: Script de validación adaptado

### Ejecución

```bash
# Entrenamiento
python p2_3_train.py

# Validación
python p2_3_validate.py
```

### Resultados Esperados

Con esta estrategia, el robot debería:
1. Detectar el obstáculo mediante sensores IR
2. Realizar maniobras de evasión (giros)
3. Continuar navegando hacia el cilindro usando información de distancia
4. Alcanzar el objetivo exitosamente
