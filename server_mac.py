#!/usr/bin/env python3
"""
Utensil Detection Server - Runs on your Mac/PC
Receives frames from Raspberry Pi, processes them, and sends back results
"""

import socket
import pickle
import struct
import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np

# Fix for PyTorch 2.6+ compatibility with YOLOv8
# Add all necessary classes to safe globals
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
    pass

# Classes that represent utensils and hands in COCO dataset
UTENSIL_CLASSES = {
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'cup': 47,
    'bottle': 39,
    'wine glass': 46
}

def process_frame(frame, model, pose_model):
    """
    Process a frame and return detection results
    """
    # Run YOLOv8 object detection for utensils
    results = model(frame, conf=0.5, verbose=False)
    
    # Run YOLOv8 pose estimation for hand detection
    pose_results = pose_model(frame, conf=0.5, verbose=False)
    
    # Get results
    result = results[0]
    pose_result = pose_results[0]
    
    # Extract utensil detections
    utensil_detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        
        if class_name in UTENSIL_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0]
            utensil_detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
    
    # Extract hand/person detections
    hand_detections = []
    keypoints_data = []
    
    for box in pose_result.boxes:
        class_id = int(box.cls[0])
        class_name = pose_result.names[class_id]
        confidence = float(box.conf[0])
        
        if class_name == 'person':
            x1, y1, x2, y2 = box.xyxy[0]
            hand_detections.append({
                'class_name': 'person',
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
    
    # Extract keypoints if available
    if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
        for keypoints in pose_result.keypoints:
            if keypoints.xy is not None:
                kpts = keypoints.xy[0].cpu().numpy()
                if len(kpts) > 10:
                    # Left wrist (index 9) and right wrist (index 10)
                    keypoints_data.append({
                        'left_wrist': [int(kpts[9][0]), int(kpts[9][1])] if kpts[9][0] > 0 else None,
                        'right_wrist': [int(kpts[10][0]), int(kpts[10][1])] if kpts[10][0] > 0 else None
                    })
    
    return {
        'utensils': utensil_detections,
        'hands': hand_detections,
        'keypoints': keypoints_data
    }

def main():
    # Server configuration
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 8888
    
    print("=" * 60)
    print("UTENSIL DETECTION SERVER - Running on Mac/PC")
    print("=" * 60)
    
    # Load models
    print("\n[1/3] Loading YOLOv8 object detection model...")
    model = YOLO('yolov8n.pt')
    print("✓ Object detection model loaded")
    
    print("[2/3] Loading YOLOv8 pose estimation model...")
    pose_model = YOLO('yolov8n-pose.pt')
    print("✓ Pose estimation model loaded")
    
    # Create socket
    print(f"[3/3] Starting server on {HOST}:{PORT}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"✓ Server listening on port {PORT}")
    
    # Get local IP address for display
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    
    print("\n" + "=" * 60)
    print("SERVER READY!")
    print("=" * 60)
    print(f"\n📡 Waiting for Raspberry Pi to connect...")
    print(f"\n💡 On your Raspberry Pi, use this IP address: {local_ip}")
    print(f"   Run: python3 pi_client.py --server {local_ip}\n")
    
    while True:
        try:
            # Accept connection
            client_socket, addr = server_socket.accept()
            print(f"\n✓ Connected to Raspberry Pi at {addr[0]}:{addr[1]}")
            
            data = b""
            payload_size = struct.calcsize("Q")
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Retrieve message size
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < payload_size:
                    break
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                # Retrieve frame data
                while len(data) < msg_size:
                    data += client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize frame
                frame = pickle.loads(frame_data)
                
                # Process frame
                detection_start = time.time()
                results = process_frame(frame, model, pose_model)
                detection_time = time.time() - detection_start
                
                # Send results back
                result_data = pickle.dumps(results)
                message_size = struct.pack("Q", len(result_data))
                client_socket.sendall(message_size + result_data)
                
                frame_count += 1
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 Processed {frame_count} frames | "
                          f"FPS: {fps:.1f} | "
                          f"Detection time: {detection_time*1000:.1f}ms | "
                          f"Utensils: {len(results['utensils'])} | "
                          f"Hands: {len(results['hands'])}")
        
        except Exception as e:
            print(f"\n⚠️  Connection error: {e}")
            print("Waiting for new connection...")
            continue
        finally:
            if 'client_socket' in locals():
                client_socket.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
