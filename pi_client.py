#!/usr/bin/env python3
"""
Utensil Detection Client - Runs on Raspberry Pi
Captures video, sends to server, receives and displays results with voice alerts
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import argparse
import time
from collections import defaultdict
import pyttsx3

# Classes for reference
UTENSIL_CLASSES = ['fork', 'knife', 'spoon', 'bowl', 'cup', 'bottle', 'wine glass']

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Client for Utensil Detection')
    parser.add_argument('--server', type=str, required=True, help='Server IP address (your Mac)')
    parser.add_argument('--port', type=int, default=8888, help='Server port (default: 8888)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RASPBERRY PI CLIENT - Utensil Detection")
    print("=" * 60)
    
    # Initialize text-to-speech engine
    print("\n[1/4] Initializing text-to-speech engine...")
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        print("✓ TTS engine initialized")
    except Exception as e:
        print(f"⚠️  TTS not available: {e}")
        tts_engine = None
    
    # Connect to server
    print(f"[2/4] Connecting to server at {args.server}:{args.port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((args.server, args.port))
        print(f"✓ Connected to server")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print(f"\nMake sure:")
        print(f"  1. Server is running on your Mac")
        print(f"  2. IP address is correct: {args.server}")
        print(f"  3. Both devices are on the same network")
        return
    
    # Open camera
    print(f"[3/4] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Raspberry Pi camera brightness/exposure fixes
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_GAIN, 10)
    
    print("✓ Camera opened")
    
    # Tracking variables for unattended utensils
    utensil_first_seen = {}
    last_alert_time = {}
    UNATTENDED_THRESHOLD = 2.0  # seconds
    ALERT_COOLDOWN = 5.0  # seconds
    
    print("[4/4] Starting detection...")
    print("\n" + "=" * 60)
    print("DETECTION ACTIVE - Press 'q' to quit")
    print("=" * 60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Apply software brightness adjustment
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
            
            # Serialize frame
            data = pickle.dumps(frame)
            message_size = struct.pack("Q", len(data))
            
            # Send frame to server
            try:
                client_socket.sendall(message_size + data)
            except Exception as e:
                print(f"Error sending frame: {e}")
                break
            
            # Receive results from server
            try:
                data = b""
                payload_size = struct.calcsize("Q")
                
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += client_socket.recv(4096)
                
                result_data = data[:msg_size]
                results = pickle.loads(result_data)
                
            except Exception as e:
                print(f"Error receiving results: {e}")
                break
            
            # Process results and draw on frame
            utensil_count = {}
            
            # Draw utensil detections (green boxes)
            for detection in results['utensils']:
                class_name = detection['class_name']
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']
                
                utensil_count[class_name] = utensil_count.get(class_name, 0) + 1
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw hand detections (blue boxes)
            hand_count = len(results['hands'])
            for detection in results['hands']:
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                label = f"Hand/Person: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw keypoints (yellow dots for wrists)
            for kp_data in results['keypoints']:
                if kp_data['left_wrist']:
                    cv2.circle(frame, tuple(kp_data['left_wrist']), 5, (0, 255, 255), -1)
                if kp_data['right_wrist']:
                    cv2.circle(frame, tuple(kp_data['right_wrist']), 5, (0, 255, 255), -1)
            
            # Display counts
            y_offset = 30
            cv2.putText(frame, "Detected Items:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            
            if hand_count > 0:
                cv2.putText(frame, f"Hands/People: {hand_count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += 25
            
            for utensil, count in utensil_count.items():
                cv2.putText(frame, f"{utensil}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Determine state and handle alerts
            current_time = time.time()
            has_utensils = len(utensil_count) > 0
            has_hands = hand_count > 0
            
            # Three states
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
            else:
                state = "NOTHING"
                state_color = (128, 128, 128)
                state_symbol = "○"
            
            # Track unattended utensils
            for utensil_type in utensil_count.keys():
                if state == "UNATTENDED":
                    if utensil_type not in utensil_first_seen:
                        utensil_first_seen[utensil_type] = current_time
                    else:
                        unattended_duration = current_time - utensil_first_seen[utensil_type]
                        
                        if unattended_duration >= UNATTENDED_THRESHOLD:
                            if (utensil_type not in last_alert_time or 
                                current_time - last_alert_time[utensil_type] >= ALERT_COOLDOWN):
                                
                                alert_message = f"Warning! {utensil_type} left unattended in sink!"
                                print(f"🔊 ALERT: {alert_message}")
                                
                                if tts_engine:
                                    try:
                                        tts_engine.say(alert_message)
                                        tts_engine.runAndWait()
                                    except:
                                        pass
                                
                                last_alert_time[utensil_type] = current_time
                            
                            warning_text = f"⚠️ {utensil_type.upper()} UNATTENDED!"
                            cv2.putText(frame, warning_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_offset += 30
                else:
                    if utensil_type in utensil_first_seen:
                        del utensil_first_seen[utensil_type]
            
            # Remove utensils no longer detected
            utensils_to_remove = [u for u in utensil_first_seen.keys() if u not in utensil_count]
            for utensil_type in utensils_to_remove:
                del utensil_first_seen[utensil_type]
            
            # Display status
            y_offset += 10
            status_text = f"STATUS: {state} {state_symbol}"
            cv2.putText(frame, status_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            
            # Show FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 FPS: {fps:.1f} | Utensils: {len(utensil_count)} | Hands: {hand_count} | State: {state}")
            
            # Display frame
            cv2.imshow('Utensil & Hand Detection (Pi Client)', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        client_socket.close()
        if tts_engine:
            try:
                tts_engine.stop()
            except:
                pass
        print("\n✓ Client stopped")

if __name__ == "__main__":
    main()
