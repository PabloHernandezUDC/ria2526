<style>
@media print {
  body {
    margin: 1cm 1.5cm;
    font-size: 12pt;
    line-height: 1.4;
  }
  h1 {
    font-size: 16pt;
    margin-top: 0.5cm;
    margin-bottom: 0.3cm;
  }
  h2 {
    font-size: 13pt;
    margin-top: 0.4cm;
    margin-bottom: 0.3cm;
  }
  h3 {
    font-size: 11pt;
    margin-top: 0.3cm;
    margin-bottom: 0.2cm;
  }
  p {
    margin-top: 0.2cm;
    margin-bottom: 0.2cm;
  }
  table {
    margin-top: 0.2cm;
    margin-bottom: 0.2cm;
  }
  @page {
    size: A4;
    margin: 1cm 1.5cm;
  }
}
</style>

# PRÁCTICA 03: DETECCIÓN EN TIEMPO REAL

## Robótica Inteligente y Autónoma (RIA) - Curso 2025-2026

**Autores:**  

- Pablo Hernández Martínez (pablo.hernandez.martinez@udc.es)
- Iván Moure Pérez (i.moure@udc.es)

---

## 1. INTRODUCCIÓN Y ARQUITECTURA DEL SISTEMA

### 1.1 Objetivo

Esta práctica implementa un sistema de control del robot Robobo real mediante detección de poses corporales en tiempo real usando modelos YOLO. El sistema opera en dos fases secuenciales: primero, una fase de **teleoperación** donde el usuario controla el robot mediante gestos con los brazos para aproximarlo hacia un objetivo; posteriormente, una fase de **aproximación autónoma** donde la política de RL aprendida en la P.01 toma el control para completar la tarea: llegar al objetivo.

### 1.2 Arquitectura del Sistema

El sistema integra múltiples componentes de visión por computador y control robótico:

<table style="width:100%; border:none;">
<tr>
<td style="width:50%; vertical-align:top; border:none;">

**Componentes principales:**

| Componente | Tecnología |
|------------|------------|
| Detección de poses | YOLOv8n-pose |
| Detección de objetos | YOLOv8n |
| Política RL | PPO (Práctica 01) |
| Streaming vídeo | robobopy_videostream |
| Control robot | robobopy |

</td>
<td style="width:50%; vertical-align:top; border:none; padding-left:20px;">

**Flujo de datos:**
La webcam del PC captura al usuario y alimenta YOLOv8-pose para clasificar sus gestos, mientras que la cámara del robot alimenta YOLOv8 para detectar el objeto objetivo. Los gestos clasificados se traducen en comandos de control para los motores del Robobo, y cuando se detecta el objetivo, se activa automáticamente la política de RL aprendida.

</td>
</tr>
</table>

### 1.3 Sistema de Control Gestual

La detección de poses considera algunos de los 17 keypoints corporales del modelo YOLOv8-pose. El sistema de control se basa en la posición relativa de las muñecas respecto a los hombros:

<table style="width:100%; border:none;">
<tr>
<td style="width:40%; vertical-align:top; border:none;">

| Gesto | Condición | Acción Robot |
|-------|-----------|--------------|
| Ambos brazos arriba | `wrist_y < shoulder_y - 20` (ambos) | Avanzar (20, 20) |
| Brazo derecho arriba | Solo derecho elevado | Girar izquierda (0, 25) |
| Brazo izquierdo arriba | Solo izquierdo elevado | Girar derecha (25, 0) |
| Sin brazos arriba | Ninguno elevado | Detener motores |

</td>
<td style="width:60%; vertical-align:top; border:none; padding-left:20px;">

**Decisiones de diseño:** El umbral de 20 píxeles entre muñeca y hombro proporciona robustez ante ruido en la detección sin requerir movimientos excesivamente exagerados. La inversión del control (brazo derecho → giro izquierda) resulta más intuitiva para el operador, ya que el robot gira hacia donde "apunta" el brazo elevado desde la perspectiva del usuario frente a la cámara. El sistema solo ejecuta acciones cuando detecta cambios de gesto, evitando comandos redundantes.

</td>
</tr>
</table>

---

## 2. MODOS DE DETECCIÓN Y TRANSICIÓN A POLÍTICA RL

### 2.1 Detección del Objeto Objetivo

El sistema soporta dos modos de detección configurables mediante la variable `MODE`:

**Modo BLOB (simulador):** Utiliza la detección nativa de blobs de color del Robobo para identificar objetos rojos. Ideal para pruebas en `RoboboSim` con un escenario de cilindro rojo.

**Modo YOLO (robot real):** Emplea YOLOv8n para detectar objetos arbitrarios definidos por la variable `TARGET` (por defecto "bottle"). La cámara del robot transmite vídeo mediante streaming, que se procesa frame a frame para localizar el objeto con un umbral de confianza de 0.5.

### 2.2 Integración con Política de Refuerzo

<table style="width:100%; border:none;">
<tr>
<td style="width:55%; vertical-align:top; border:none;">

**Condiciones de activación:** La política RL se activa automáticamente cuando: (1) el objeto objetivo es detectado por la cámara del robot, y (2) el usuario ha completado al menos 5 acciones de teleoperación.

**Adaptación de observaciones:** La política PPO entrenada en la P.01 espera observaciones discretizadas en 6 sectores visuales. Para compatibilidad, las coordenadas del objeto detectado por YOLO se normalizan al rango [0, 100] y se mapean a sectores mediante `sector = target_x // 20`, manteniendo el sector 5 para cuando el objetivo no es visible.

</td>
<td style="width:45%; vertical-align:top; border:none; padding-left:20px;">

**Ejecución de la política:**

```python
observation = {"sector": np.array([sector])}
action, _ = model.predict(obs, deterministic=True)

# Acciones: 0=Avanzar, 1=Izquierda, 2=Derecha
if action == 0:
    rob.moveWheelsByTime(10, 10, 0.5)
elif action == 1:
    rob.moveWheelsByTime(10, 0, 0.5)
elif action == 2:
    rob.moveWheelsByTime(0, 10, 0.5)
```

**Condición de éxito:** El episodio finaliza cuando el área del objeto detectado supera 70000 píxeles (modo YOLO) o el blob supera 8000 con posición centrada (modo BLOB).

</td>
</tr>
</table>

---

## 3. RESULTADOS Y CONCLUSIONES

### 3.1 Demostración del sistema

El funcionamiento del sistema se demuestra en dos vídeos complementarios. El primero (`pov_ordenador.mp4`) consiste en una grabación de pantalla del ordenador que muestra simultáneamente las dos cámaras activas durante la ejecución: la cámara del móvil montado sobre el Robobo, que transmite la perspectiva del robot en tiempo real, y la webcam del ordenador donde el usuario realiza los gestos de control para dirigir el movimiento del robot según su criterio. Este vídeo permite observar la interfaz de detección de poses con las anotaciones de YOLOv8-pose, el feedback visual de las acciones detectadas, la ventana de la cámara del robot con la detección YOLO del objeto objetivo, y la activación automática de la política de refuerzo tras completar las acciones mínimas de teleoperación.

El segundo vídeo (`pov_persona.mp4`) ofrece una grabación desde otro punto de vista que captura el comportamiento físico del Robobo durante la ejecución. Se puede observar de nuevo al robot navegando hacia el objetivo primero bajo el control del usuario y posteriormente de forma autónoma con RL.

### 3.2 Análisis del sistema

Entre las **fortalezas** del sistema destaca la integración fluida entre los componentes de visión artificial basados en YOLO y el control robótico, permitiendo una operación intuitiva mediante gestos naturales. La reutilización de la política de RL aprendida en la P.01 demuestra la transferibilidad del aprendizaje entre contextos, mientras que la arquitectura modular con modos BLOB y YOLO intercambiables facilita tanto el desarrollo en simulador como el despliegue en el robot real. El feedback visual en tiempo real proporciona al operador información continua sobre el estado del sistema.

Respecto a las **limitaciones**, la latencia inherente al streaming de vídeo desde el móvil puede introducir retrasos perceptibles en la respuesta del robot ante los comandos. La detección de poses muestra sensibilidad a las condiciones de iluminación del entorno, y el umbral fijo de confianza de 0.5 podría requerir ajustes según las características específicas de cada escenario. Además, la política de RL entrenada exclusivamente en simulador puede manifestar el fenómeno *reality gap* al ejecutarse en el robot físico.

### 3.3 Conclusiones

Este trabajo demuestra la viabilidad de sistemas híbridos que combinan teleoperación gestual con control autónomo. La arquitectura modular permite alternar entre control humano y política de refuerzo de forma transparente, aprovechando las fortalezas de cada modo: la intuición humana para navegación de largo alcance y la precisión del agente entrenado para la aproximación final. La integración de YOLOv8 para detección de poses y objetos proporciona una base robusta y extensible para aplicaciones de robótica interactiva en tiempo real.
