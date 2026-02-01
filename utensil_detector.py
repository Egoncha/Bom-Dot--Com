
import cv2
import torch
from ultralytics import YOLO
import time
from collections import defaultdict
import pyttsx3
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.batchnorm import BatchNorm2d
    from torch.nn.modules.activation import SiLU
    import collections
    
    safe_globals = [
        DetectionModel,
        Sequential,
        Conv2d,
        BatchNorm2d,
        SiLU,
        collections.OrderedDict
    ]
    
    torch.serialization.add_safe_globals(safe_globals)
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")
    print("Attempting to use older loading method...")
    pass

UTENSIL_CLASSES = {
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'cup': 47,
    'bottle': 39,
    'wine glass': 46
}

HAND_CLASS = 'person'

def main():
    print("Initializing text-to-speech engine...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Load YOLOv8 model (will download automatically on first run)
    print("Loading YOLOv8 object detection model...")
    model = YOLO('yolov8n.pt')  # 'n' for nano (fastest), can use 's', 'm', 'l', 'x' for larger models
    
    # Load YOLOv8 pose model for hand/person detection
    print("Loading YOLOv8 pose estimation model...")
    pose_model = YOLO('yolov8n-pose.pt')  # Detects people and their keypoints including hands
    
    # Tracking variables for unattended utensils
    utensil_first_seen = {}  # Dictionary to track when each utensil was first seen unattended
    last_alert_time = {}  # Track when we last alerted for each utensil type
    UNATTENDED_THRESHOLD = 2.0  # seconds
    ALERT_COOLDOWN = 5.0  # seconds between alerts for same utensil type
    
    # Open webcam (0 is default camera, change to 1, 2, etc. if you have multiple cameras)
    print("Opening webcam...")
    
    # Try different camera backends for macOS compatibility
    cap = None
    for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Camera opened successfully with backend: {backend}")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open webcam")
        print("\nTroubleshooting steps:")
        print("1. Go to System Settings > Privacy & Security > Camera")
        print("2. Enable camera access for Terminal/Python")
        print("3. Close and restart Terminal")
        print("4. Make sure no other app is using the camera")
        return
    
    # Set camera properties (optional) - some may fail on macOS, that's OK
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except:
        pass  # Continue even if setting properties fails
    
    print("Starting detection... Press 'q' to quit")
    
    # Sometimes the first few frames fail on macOS, so retry
    frame_count = 0
    consecutive_failures = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:  # 30 consecutive failures
                print("Error: Too many failed frame captures")
                break
            continue  # Try next frame
        
        consecutive_failures = 0  # Reset on success
        frame_count += 1
        
        # Run YOLOv8 object detection for utensils
        results = model(frame, conf=0.5, verbose=False)  # conf is confidence threshold
        
        # Run YOLOv8 pose estimation for hand detection
        pose_results = pose_model(frame, conf=0.5, verbose=False)
        
        # Get the first result (we only have one frame)
        result = results[0]
        pose_result = pose_results[0]
        
        # Count utensils and hands detected
        utensil_count = {}
        hand_count = 0
        
        # Draw bounding boxes and labels
        for box in result.boxes:
            # Get class ID and name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Check if it's a utensil
            if class_name in UTENSIL_CLASSES:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Count this utensil
                utensil_count[class_name] = utensil_count.get(class_name, 0) + 1
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Background for text
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Put text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw hand/person detections from pose model
        for box in pose_result.boxes:
            class_id = int(box.cls[0])
            class_name = pose_result.names[class_id]
            confidence = float(box.conf[0])
            
            if class_name == 'person':  # Detect people (hands/arms)
                hand_count += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle in blue for hands/people
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw label
                label = f"Hand/Person: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Background for text
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (255, 0, 0), -1)
                
                # Put text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Optionally draw hand keypoints if available
        if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
            for keypoints in pose_result.keypoints:
                if keypoints.xy is not None:
                    # Draw wrist, left hand, right hand keypoints (indices 9, 10 in COCO pose)
                    kpts = keypoints.xy[0].cpu().numpy()
                    if len(kpts) > 10:
                        # Left wrist
                        if kpts[9][0] > 0 and kpts[9][1] > 0:
                            cv2.circle(frame, (int(kpts[9][0]), int(kpts[9][1])), 5, (0, 255, 255), -1)
                        # Right wrist
                        if kpts[10][0] > 0 and kpts[10][1] > 0:
                            cv2.circle(frame, (int(kpts[10][0]), int(kpts[10][1])), 5, (0, 255, 255), -1)
        
        # Display count summary
        y_offset = 30
        cv2.putText(frame, "Detected Items:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Show hand count first
        if hand_count > 0:
            cv2.putText(frame, f"Hands/People: {hand_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
        
        # Show utensil counts
        for utensil, count in utensil_count.items():
            cv2.putText(frame, f"{utensil}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Check for unattended utensils
        current_time = time.time()
        has_utensils = len(utensil_count) > 0
        has_hands = hand_count > 0
        
        # Determine the state (3 states)
        if not has_utensils and not has_hands:
            state = "NOTHING"
            state_color = (128, 128, 128)  # Gray
            state_symbol = "○"
        elif has_utensils and not has_hands:
            state = "UNATTENDED"
            state_color = (0, 165, 255)  # Orange
            state_symbol = "⚠️"
        elif has_utensils and has_hands:
            state = "ATTENDED"
            state_color = (0, 255, 0)  # Green
            state_symbol = "✓"
        else:  # has_hands but not has_utensils (person present, no utensils)
            state = "NOTHING"
            state_color = (128, 128, 128)  # Gray
            state_symbol = "○"
        
        # Track each type of utensil
        for utensil_type in utensil_count.keys():
            if state == "UNATTENDED":
                # Utensil is unattended
                if utensil_type not in utensil_first_seen:
                    # First time seeing this utensil unattended
                    utensil_first_seen[utensil_type] = current_time
                else:
                    # Check how long it's been unattended
                    unattended_duration = current_time - utensil_first_seen[utensil_type]
                    
                    if unattended_duration >= UNATTENDED_THRESHOLD:
                        # Check if we should alert (not alerted recently)
                        if (utensil_type not in last_alert_time or 
                            current_time - last_alert_time[utensil_type] >= ALERT_COOLDOWN):
                            
                            # TRIGGER VOICE ALERT
                            alert_message = f"Warning! {utensil_type} left unattended in sink!"
                            print(f"🔊 ALERT: {alert_message}")
                            
                            # Play voice alert (non-blocking)
                            try:
                                tts_engine.say(alert_message)
                                tts_engine.runAndWait()
                            except:
                                pass  # Continue even if TTS fails
                            
                            last_alert_time[utensil_type] = current_time
                        
                        # Show warning on screen
                        warning_text = f"⚠️ {utensil_type.upper()} UNATTENDED!"
                        cv2.putText(frame, warning_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        y_offset += 30
            else:
                # Utensil is attended or no longer present, reset timer
                if utensil_type in utensil_first_seen:
                    del utensil_first_seen[utensil_type]
        
        # Remove utensils that are no longer detected
        utensils_to_remove = [u for u in utensil_first_seen.keys() if u not in utensil_count]
        for utensil_type in utensils_to_remove:
            del utensil_first_seen[utensil_type]
        
        # Display attendance status
        y_offset += 10
        status_text = f"STATUS: {state} {state_symbol}"
        cv2.putText(frame, status_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Show frame
        cv2.imshow('Utensil & Hand Detection - Press Q to quit', frame)
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    try:
        tts_engine.stop()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()
