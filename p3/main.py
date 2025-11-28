import time
from datetime import datetime

import cv2
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobopy_videostream.RoboboVideo import RoboboVideo
from robobosim.RoboboSim import RoboboSim
from stable_baselines3 import PPO
from ultralytics import YOLO

# --------------------

# "blob" o "yolo"
MODE = "yolo"

# clase objetivo para YOLO
TARGET = "bottle"  # Clase de objeto objetivo para detección YOLO

# IP del robot (localhost para simulador)
IP = "172.20.10.3"

# --------------------

# Índices de keypoints de YOLOv8-pose
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}


def get_angle(p1, p2, p3):
    """
    Calcula el ángulo entre tres puntos.
    p2 es el vértice del ángulo.
    Devuelve el ángulo en grados.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calcular ángulo usando producto escalar
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def classify_pose(keypoints, confidence_threshold=0.5):
    """
    Clasifica la pose basándose en las posiciones de los keypoints.
    Devuelve la acción de pose detectada para control del robot.
    """
    if keypoints is None or len(keypoints) == 0:
        return "Neutral"
    
    # Extraer keypoints con verificación de confianza
    kp = {}
    for name, idx in KEYPOINT_DICT.items():
        if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
            kp[name] = keypoints[idx][:2]  # (x, y)
        else:
            kp[name] = None
    
    # Esquema de control:
    # - Ambos brazos arriba: Avanzar
    # - Brazo derecho levantado: Girar derecha
    # - Brazo izquierdo levantado: Girar izquierda
    # - Sin brazos levantados: Detenerse
    
    left_arm_raised = False
    right_arm_raised = False
    
    # Verificar si el brazo derecho está levantado
    if kp.get("right_shoulder") is not None and kp.get("right_wrist") is not None:
        if kp["right_wrist"][1] < kp["right_shoulder"][1] - 20:
            right_arm_raised = True
    
    # Verificar si el brazo izquierdo está levantado
    if kp.get("left_shoulder") is not None and kp.get("left_wrist") is not None:
        if kp["left_wrist"][1] < kp["left_shoulder"][1] - 20:
            left_arm_raised = True
    
    # Determinar acción basada en posición de brazos
    if left_arm_raised and right_arm_raised:
        return "Forward"
    elif right_arm_raised:
        return "Left"
    elif left_arm_raised:
        return "Right"
    else:
        return "Stop"


def control_robot(action, rob, last_action, action_count):
    """
    Controla el robot basándose en la acción de pose detectada.
    Devuelve last_action y action_count actualizados.
    """
    # Solo ejecutar acción si es diferente a la anterior
    if action != last_action:
        if action == "Forward":
            print(f"[ACTION {action_count}] Moving FORWARD")
            rob.moveWheels(20, 20)  # Avanzar
        elif action == "Left":
            print(f"[ACTION {action_count}] Turning LEFT")
            rob.moveWheels(0, 25)  # Girar izquierda
        elif action == "Right":
            print(f"[ACTION {action_count}] Turning RIGHT")
            rob.moveWheels(25, 0)  # Girar derecha
        elif action == "Stop":
            print(f"[ACTION {action_count}] STOPPING")
            rob.stopMotors()
        
        action_count += 1
        last_action = action
    
    return last_action, action_count


def detect_red_blob(rob):
    """
    Detecta blob rojo usando la detección de blob de color de Robobo.
    Devuelve True si se detecta el blob, False en caso contrario.
    """
    blob = rob.readColorBlob(BlobColor.RED)
    if blob and blob.size > 100:  # Umbral mínimo de tamaño
        return True, blob
    return False, None


def detect_yolo_object(video_stream, yolo_model, target_class="bottle", confidence_threshold=0.5):
    """
    Detecta objeto objetivo usando modelo YOLO con la cámara del robot.
    Devuelve True si se detecta el objeto con suficiente confianza, False en caso contrario.
    """
    # Obtener frame de la cámara del robot
    frame, timestamp, sync_id, frame_id = video_stream.getImageWithMetadata()
    
    if frame is None:
        return False, None, None
    
    # Ejecutar detección YOLO
    results = yolo_model(frame, verbose=False)
    
    # Buscar objeto objetivo
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener nombre de clase y confianza
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            
            # Verificar si es el objeto objetivo con suficiente confianza
            if class_name == target_class and conf >= confidence_threshold:
                # Obtener coordenadas del cuadro
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Dibujar cuadro delimitador en el frame
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dibujar etiqueta con nombre de clase y confianza
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Dibujar coordenadas en la parte inferior del frame
                coords_text = f"Center: ({box_center_x:.0f}, {(y1 + y2) / 2:.0f}) | Area: {box_area:.0f}"
                cv2.putText(frame, coords_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detection_info = {
                    "confidence": conf,
                    "center_x": box_center_x,
                    "area": box_area,
                    "width": box_width,
                    "height": box_height
                }
                
                return True, detection_info, frame
    
    return False, None, frame


def run_rl_policy_step(rob, model, mode="blob", target_color=BlobColor.RED, video_stream=None, object_model=None, target_class="bottle"):
    """
    Ejecuta un paso de la política de RL.
    Devuelve tupla (continuar, target_x, target_size, sector, acción).
    """
    if mode == "blob":
        # Leer información del blob
        blob = rob.readColorBlob(target_color)
        
        if blob is None or blob.size < 50:
            rob.moveWheels(15, 5)  # Girar buscando el blob
            return True, None, None, None, None
        
        # Usar coordenadas del blob (ya normalizadas en rango 0-100)
        target_x = blob.posx
        target_size = blob.size
        
    else:  # modo == "yolo"
        time.sleep(0.3)
        # Detectar objeto con YOLO
        yolo_detected, yolo_info, robot_frame = detect_yolo_object(video_stream, object_model, target_class, confidence_threshold=0.2)
        
        if not yolo_detected:
            rob.moveWheels(5, 0)  # Girar buscando el objeto
            return True, None, None, None, None
        
        # Normalizar coordenadas al rango 0-100
        # Asumiendo que robot_frame tiene dimensiones estándar
        frame_width = robot_frame.shape[1]
        frame_center = frame_width / 2
        
        # Normalizar center_x: centro de la imagen = 50, izquierda = 0, derecha = 100
        target_x = ((yolo_info['center_x'] - frame_center) / frame_center) * 50 + 50
        target_x = np.clip(target_x, 0, 100)  # Asegurar que esté en rango 0-100
        
        target_size = yolo_info['area']
    
    # Crear observación compatible con el modelo de la práctica 01
    # El modelo espera un diccionario con clave "sector" (0-5)
    if target_x == 0:
        sector = 5
    elif target_x == 100:
        sector = 4
    else:
        sector = int(target_x // 20)
    
    observation = {
        "sector": np.array([sector], dtype=int).flatten()
    }
    
    # Predecir acción con el modelo
    action, _states = model.predict(observation, deterministic=True)
    
    # Ejecutar acción
    if action == 0:  # Avanzar
        rob.moveWheelsByTime(10, 10, 0.5)
    elif action == 1:  # Girar izquierda
        rob.moveWheelsByTime(10, 0, 0.5)
    elif action == 2:  # Girar derecha
        rob.moveWheelsByTime(0, 10, 0.5)
    
    # Condición de éxito: objeto grande y centrado
    if mode == "blob":
        success_condition = target_size > 8000 and abs(target_x - 50) < 20
    else:
        success_condition = target_size > 70000
    
    return not success_condition, target_x, target_size, sector, action


def main():
    # Cargar el modelo YOLOv8 de poses
    pose_model = YOLO("yolo_models/yolov8n-pose.pt")
    
    # Cargar modelo YOLO de detección de objetos si se usa modo YOLO
    if MODE == "yolo":
        object_model = None
        video_stream = None
        print("Cargando modelo YOLO para detección de objetos...")
        object_model = YOLO("yolo_models/yolov8n.pt")  # Modelo YOLOv8 estándar para detección de objetos
        print("Modelo YOLO cargado correctamente")
    
    # Cargar modelo de RL de la práctica 01
    print("Cargando modelo de RL de práctica 01...")
    rl_model = PPO.load("checkpoints/1/checkpoint.zip")
    print("Modelo cargado correctamente")
    
    # Conectar con RoboboSim
    print("Conectando con Robobo...")
    rob = Robobo(IP)
    rob.connect()

    if MODE == "blob":
        sim = RoboboSim(IP)
        sim.connect()
    print("Conectado correctamente al simulador")
    
    rob.moveTiltTo(100, speed=10, wait=True)

    
    # Inicializar stream de video si se usa modo YOLO
    if MODE == "yolo":
        print("Inicializando stream de video del robot...")
        video_stream = RoboboVideo(IP)
        video_stream.connect()
        rob.startStream()
        print("Stream de video inicializado correctamente")
    
    # Usar la webcam del ordenador (0 = cámara predeterminada)
    print("Abriendo cámara del ordenador...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened() and MODE == "blob":
        print("Error: No se pudo abrir la cámara")
        rob.disconnect()
        sim.disconnect()
        return
    
    print("Cámara abierta correctamente")
    print("\n=== CONTROL DEL ROBOBO ===")
    print("- Ambos brazos arriba: Avanzar")
    print("- Brazo derecho arriba: Girar derecha")
    print("- Brazo izquierdo arriba: Girar izquierda")
    print("- Sin brazos: Detener")
    if MODE == "blob":
        print(f"\nModo: BLOB DETECTION (Blob rojo)")
        print("Cuando se detecte el BLOB ROJO y hayas hecho al menos 5 acciones,")
    else:
        print(f"\nModo: YOLO OBJECT DETECTION (Objeto: {TARGET})")
        print(f"Cuando se detecte '{TARGET}' y hayas hecho al menos 5 acciones,")
    print("se activará automáticamente la política de refuerzo.")
    print("Presiona 'q' para salir\n")
    
    last_action = None
    action_count = 0
    rl_activated = False
    rl_step = 0
    max_rl_steps = 1000
    min_actions = 5  # Mínimo de acciones antes de activar RL
    
    try:
        while True:
            # Capturar frame desde la cámara
            ret, frame = cap.read()
            
            if not ret:
                print("Error: No se pudo capturar imagen")
                break
            
            # Ejecutar detección de poses YOLOv8 en el frame
            results = pose_model(frame, verbose=False)
            
            # Obtener marca de tiempo actual
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Visualizar los resultados en el frame
            annotated_frame = results[0].plot()
            
            # Verificar detección según el modo seleccionado
            target_detected = False
            detection_info = None
            
            if MODE == "blob":
                # Modo blob: detectar blob rojo
                blob_detected, blob = detect_red_blob(rob)
                if blob_detected:
                    target_detected = True
                    detection_info = f"size: {blob.size:.0f}"
                    cv2.putText(annotated_frame, f"RED BLOB DETECTED! Size: {blob.size:.0f}", 
                               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
            else:
                # Modo YOLO: detectar objeto objetivo
                yolo_detected, yolo_info, robot_frame = detect_yolo_object(video_stream, object_model, TARGET, confidence_threshold=0.5)
                if yolo_detected:
                    target_detected = True
                    detection_info = f'conf: {yolo_info["confidence"]:.2f}, area: {yolo_info["area"]:.0f}'
                
                # Mostrar feed de la cámara del robot en modo YOLO
                if robot_frame is not None:
                    cv2.imshow("Robot Camera", robot_frame)
            
            # Manejar activación y ejecución de RL
            if not rl_activated:
                if target_detected and action_count >= min_actions:
                    if MODE == "blob":
                        print(f"\n🔴 ¡BLOB ROJO DETECTADO! ({detection_info})")
                    else:
                        print(f"\n🎯 ¡{TARGET.upper()} DETECTADO! ({detection_info})")
                    print(f"✅ Acciones completadas: {action_count} >= {min_actions}")
                    print("\n" + "="*50)
                    print(" ACTIVANDO POLÍTICA DE REFUERZO (Práctica 01)")
                    print("="*50)
                    
                    # Activar política de RL
                    rl_activated = True
                    rl_step = 0
                
                # Extraer keypoints y clasificar poses (solo primera persona)
                if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    # Solo procesar la primera persona detectada
                    keypoints = results[0].keypoints.data[0]
                    
                    # Convertir a array numpy y obtener keypoints (x, y, confianza)
                    kp_array = keypoints.cpu().numpy()
                    
                    # Clasificar la pose
                    detected_action = classify_pose(kp_array)
                    
                    # Controlar el robot basándose en la pose la pose
                    last_action, action_count = control_robot(
                        detected_action, rob, last_action, action_count
                    )
                    
                    # Imprimir marca de tiempo y acción detectada
                    print(f"[{timestamp}] {detected_action}")
                    
                    # Mostrar acción detectada en el frame
                    cv2.putText(annotated_frame, f"Action: {detected_action}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Actions: {action_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 0), 2)
            else:
                # Ejecutar un paso de RL
                if rl_step < max_rl_steps:
                    if MODE == "blob":
                        continue_rl, target_x, target_size, sector, action = run_rl_policy_step(
                            rob, rl_model, mode="blob"
                        )
                    else:
                        continue_rl, target_x, target_size, sector, action = run_rl_policy_step(
                            rob, rl_model, mode="yolo", video_stream=video_stream, 
                            object_model=object_model, target_class=TARGET
                        )
                    
                    if target_x is not None:
                        action_names = ["Avanzar", "Girar IZQUIERDA", "Girar DERECHA"]
                        action_idx = int(action) if isinstance(action, np.ndarray) else action
                        print(f"[Step {rl_step}] RL: {action_names[action_idx]} (size: {target_size:.0f}, pos: {target_x:.0f}, sector: {sector})")
                    else:
                        print(f"[Step {rl_step}] Objetivo no detectado, buscando...")
                    
                    # Mostrar estado de RL en el frame
                    cv2.putText(annotated_frame, f"RL ACTIVE - Step: {rl_step}/{max_rl_steps}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, (0, 255, 255), 2)
                    if target_x is not None:
                        cv2.putText(annotated_frame, f"Target: pos={target_x:.0f}, size={target_size:.0f}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 255), 2)
                    
                    rl_step += 1
                    
                    if not continue_rl:
                        print(f"\n✅ ¡OBJETIVO ALCANZADO! (size: {target_size:.0f}, pos: {target_x:.0f})")
                        rob.stopMotors()
                        print("="*50)
                        print("Política de refuerzo finalizada")
                        print("="*50 + "\n")
                        break
                else:
                    print("\n⏱️ Máximo de pasos alcanzado")
                    rob.stopMotors()
                    print("="*50)
                    print("Política de refuerzo finalizada")
                    print("="*50 + "\n")
                    break
            
            # Mostrar el frame resultante
            window_title = f"YOLOv8 Pose Detection + Robobo Control [{MODE.upper()} mode]"
            cv2.imshow(window_title, annotated_frame)
            
            # Salir del bucle si se presiona "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario")
    
    finally:
        # Detener motores y desconectar
        rob.stopMotors()
        if video_stream is not None:
            try:
                video_stream.disconnect()
            except:
                pass
        rob.disconnect()
        if MODE == "blob":
            sim.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Robot desconectado y ventanas cerradas")


if __name__ == "__main__":
    main()
