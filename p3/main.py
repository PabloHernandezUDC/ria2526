"""
YOLO Pose Detection from Camera Feed
Uses Ultralytics YOLOv8 for real-time pose estimation with pose classification
"""

from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime


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
    Returns a list of detected poses.
    """
    if keypoints is None or len(keypoints) == 0:
        return []
    
    poses = []
    
    # Extract keypoints with confidence check
    kp = {}
    for name, idx in KEYPOINT_DICT.items():
        if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
            kp[name] = keypoints[idx][:2]  # (x, y)
        else:
            kp[name] = None
    
    # Check for various poses
    
    # 1. Arms Up (Both arms raised above shoulders)
    if all(kp[k] is not None for k in ['left_shoulder', 'left_wrist', 'right_shoulder', 'right_wrist']):
        if kp['left_wrist'][1] < kp['left_shoulder'][1] and kp['right_wrist'][1] < kp['right_shoulder'][1]:
            poses.append("Arms Up")
    
    # 2. Right Arm Raised
    if kp.get('right_shoulder') is not None and kp.get('right_wrist') is not None:
        if kp['right_wrist'][1] < kp['right_shoulder'][1] - 50:  # Wrist significantly above shoulder
            poses.append("Right Arm Raised")
    
    # 3. Left Arm Raised
    if kp.get('left_shoulder') is not None and kp.get('left_wrist') is not None:
        if kp['left_wrist'][1] < kp['left_shoulder'][1] - 50:
            poses.append("Left Arm Raised")
    
    return poses if poses else ["Neutral"]


def main():
    # Load the YOLOv8 pose model
    model = YOLO('yolo_models/yolov8n-pose.pt')
    
    # Open the default camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting pose detection... Press 'q' to quit")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Run YOLOv8 pose detection on the frame
            results = model(frame, verbose=False)
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Extract keypoints and classify poses
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                for person_idx, keypoints in enumerate(results[0].keypoints.data):
                    # Convert to numpy array and get keypoints (x, y, confidence)
                    kp_array = keypoints.cpu().numpy()
                    
                    # Classify the pose
                    detected_poses = classify_pose(kp_array)
                    
                    # Print timestamp and detected poses
                    print(f"[{timestamp}] Person {person_idx + 1}: {', '.join(detected_poses)}")
                    
                    # Display detected poses on the frame
                    y_offset = 30 + (person_idx * 100)
                    cv2.putText(annotated_frame, f"Person {person_idx + 1}:", 
                               (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 
                               1.75, (0, 0, 0), 2)
                    
                    for i, pose in enumerate(detected_poses):
                        cv2.putText(annotated_frame, f"  - {pose}", 
                                   (10, y_offset + 25 + (i * 25)), 
                                   cv2.FONT_HERSHEY_PLAIN, 
                                   1.5, (0, 0, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    main()
