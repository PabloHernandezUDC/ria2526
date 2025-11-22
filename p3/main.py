"""
YOLO Pose Detection + Robobo Teleoperation
Uses Ultralytics YOLOv8 for real-time pose estimation to control Robobo
"""

from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from stable_baselines3 import PPO
import time


# YOLOv8-pose keypoint indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


def get_angle(p1, p2, p3):
    """
    Calculate angle between three points.
    p2 is the vertex of the angle.
    Returns angle in degrees.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def classify_pose(keypoints, confidence_threshold=0.5):
    """
    Classify the pose based on keypoint positions.
    Returns the detected pose action for robot control.
    """
    if keypoints is None or len(keypoints) == 0:
        return "Neutral"
    
    # Extract keypoints with confidence check
    kp = {}
    for name, idx in KEYPOINT_DICT.items():
        if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
            kp[name] = keypoints[idx][:2]  # (x, y)
        else:
            kp[name] = None
    
    # Control scheme:
    # - Both arms up: Move forward
    # - Right arm raised: Turn right
    # - Left arm raised: Turn left
    # - No arms raised: Stop
    
    left_arm_raised = False
    right_arm_raised = False
    
    # Check if right arm is raised
    if kp.get('right_shoulder') is not None and kp.get('right_wrist') is not None:
        if kp['right_wrist'][1] < kp['right_shoulder'][1] - 20:
            right_arm_raised = True
    
    # Check if left arm is raised
    if kp.get('left_shoulder') is not None and kp.get('left_wrist') is not None:
        if kp['left_wrist'][1] < kp['left_shoulder'][1] - 20:
            left_arm_raised = True
    
    # Determine action based on arm positions
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
    Control the robot based on detected pose action.
    Returns updated last_action and action_count.
    """
    # Solo ejecutar acción si es diferente a la anterior
    if action != last_action:
        if action == "Forward":
            print(f"[ACTION {action_count}] Moving FORWARD")
            rob.moveWheels(20, 20)  # Avanzar
        elif action == "Left":
            print(f"[ACTION {action_count}] Turning LEFT")
            rob.moveWheels(10, 20)  # Girar izquierda
        elif action == "Right":
            print(f"[ACTION {action_count}] Turning RIGHT")
            rob.moveWheels(20, 10)  # Girar derecha
        elif action == "Stop":
            print(f"[ACTION {action_count}] STOPPING")
            rob.stopMotors()
        
        action_count += 1
        last_action = action
    
    return last_action, action_count


def detect_red_blob(rob):
    """
    Detect red blob using Robobo's color blob detection.
    Returns True if blob is detected, False otherwise.
    """
    blob = rob.readColorBlob(BlobColor.RED)
    if blob and blob.size > 100:  # Umbral mínimo de tamaño
        return True, blob
    return False, None


def run_rl_policy(rob, model, target_color=BlobColor.RED):
    """
    Execute the RL policy from practice 01 to approach the red blob.
    """
    print("\n" + "="*50)
    print(" ACTIVANDO POLÍTICA DE REFUERZO (Práctica 01)")
    print("="*50)
    
    max_steps = 100
    step = 0
    
    while step < max_steps:
        # Leer información del blob
        blob = rob.readColorBlob(target_color)
        
        if blob is None or blob.size < 50:
            print(f"[Step {step}] Blob no detectado, buscando...")
            rob.moveWheels(15, 5)  # Girar buscando el blob
            time.sleep(0.5)
            step += 1
            continue
        
        # Crear observación compatible con el modelo de la práctica 01
        # El modelo espera un diccionario con clave "sector" (0-5)
        red_x = blob.posx
        if red_x == 0:
            sector = 5
        elif red_x == 100:
            sector = 4
        else:
            sector = red_x // 20
        
        observation = {
            "sector": np.array([sector], dtype=int).flatten()
        }
        
        # Predecir acción con el modelo
        action, _states = model.predict(observation, deterministic=True)
        
        # Ejecutar acción
        if action == 0:  # Avanzar
            print(f"[Step {step}] RL: Avanzando (blob size: {blob.size:.0f}, pos: {blob.posx:.0f}, sector: {sector})")
            rob.moveWheels(20, 20)
        elif action == 1:  # Girar izquierda
            print(f"[Step {step}] RL: Girando IZQUIERDA (blob pos: {blob.posx:.0f}, sector: {sector})")
            rob.moveWheels(20, 0)
        elif action == 2:  # Girar derecha
            print(f"[Step {step}] RL: Girando DERECHA (blob pos: {blob.posx:.0f}, sector: {sector})")
            rob.moveWheels(0, 20)
        
        time.sleep(0.1)
        step += 1
        
        # Condición de éxito: blob grande y centrado
        if blob.size > 8000 and abs(blob.posx) < 50:
            print(f"\n✅ ¡OBJETIVO ALCANZADO! (size: {blob.size:.0f})")
            rob.stopMotors()
            break
    
    rob.stopMotors()
    print("="*50)
    print("Política de refuerzo finalizada")
    print("="*50 + "\n")


def main():
    # Load the YOLOv8 pose model
    model = YOLO('yolo_models/yolov8n-pose.pt')
    
    # Cargar modelo de RL de la práctica 01
    print("Cargando modelo de RL de práctica 01...")
    rl_model = PPO.load("checkpoints/1/checkpoint.zip")
    print("Modelo cargado correctamente")
    
    # Conectar con RoboboSim
    print("Conectando con RoboboSim...")
    sim = RoboboSim(ip="127.0.0.1")
    rob = Robobo(ip="127.0.0.1")
    sim.connect()
    rob.connect()
    print("Conectado correctamente al simulador")
    
    # Usar la webcam del ordenador (0 = cámara predeterminada)
    print("Abriendo cámara del ordenador...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
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
    print("\nCuando se detecte el BLOB ROJO y hayas hecho al menos 5 acciones,")
    print("se activará automáticamente la política de refuerzo.")
    print("Presiona 'q' para salir\n")
    
    last_action = None
    action_count = 0
    rl_activated = False
    min_actions = 5  # Mínimo de acciones antes de activar RL
    
    try:
        while True:
            # Verificar si ya se activó la política RL
            if not rl_activated:
                # Capturar frame desde la cámara
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: No se pudo capturar imagen")
                    break
                
                # Run YOLOv8 pose detection on the frame
                results = model(frame, verbose=False)
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                # Verificar si se detecta el blob rojo
                blob_detected, blob = detect_red_blob(rob)
                
                if blob_detected and action_count >= min_actions:
                    print(f"\n🔴 ¡BLOB ROJO DETECTADO! (size: {blob.size:.0f})")
                    print(f"✅ Acciones completadas: {action_count} >= {min_actions}")
                    cv2.destroyAllWindows()
                    cap.release()
                    
                    # Activar política de RL
                    rl_activated = True
                    run_rl_policy(rob, rl_model)
                    break
                
                # Extract keypoints and classify poses
                if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    for person_idx, keypoints in enumerate(results[0].keypoints.data):
                        # Convert to numpy array and get keypoints (x, y, confidence)
                        kp_array = keypoints.cpu().numpy()
                        
                        # Classify the pose
                        detected_action = classify_pose(kp_array)
                        
                        # Control the robot based on pose
                        last_action, action_count = control_robot(
                            detected_action, rob, last_action, action_count
                        )
                        
                        # Print timestamp and detected action
                        print(f"[{timestamp}] Person {person_idx + 1}: {detected_action}")
                        
                        # Display detected action on the frame
                        y_offset = 30 + (person_idx * 60)
                        cv2.putText(annotated_frame, f"Person {person_idx + 1}: {detected_action}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1.0, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Actions: {action_count}", 
                                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 255, 0), 2)
                        
                        # Mostrar si el blob está cerca
                        if blob_detected:
                            cv2.putText(annotated_frame, f"BLOB DETECTED! Size: {blob.size:.0f}", 
                                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 0, 255), 2)
                
                # Display the resulting frame
                cv2.imshow('YOLOv8 Pose Detection + Robobo Control', annotated_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Detener motores y desconectar
        rob.stopMotors()
        rob.disconnect()
        sim.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Robot desconectado y ventanas cerradas")


if __name__ == "__main__":
    main()
