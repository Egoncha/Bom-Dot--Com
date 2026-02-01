# Client-Server Setup Guide

This setup allows your Raspberry Pi to offload heavy AI processing to your Mac/PC, achieving real-time performance.

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│  Raspberry Pi   │  WiFi/Network      │    Mac/PC       │
│   (CLIENT)      │ ─────────────────> │   (SERVER)      │
│                 │                    │                 │
│ • Captures      │ Send frames        │ • YOLOv8 models │
│   video         │ ───────────>       │ • GPU detection │
│ • Displays      │                    │ • Returns       │
│   results       │ <───────────       │   results       │
│ • Voice alerts  │ Receive results    │                 │
└─────────────────┘                    └─────────────────┘
```

## Setup Instructions

### Part 1: Setup Your Mac/PC (Server)

**1. Install dependencies on your Mac:**
```bash
pip3 install opencv-python ultralytics torch numpy
```

**2. Get your Mac's IP address:**

On Mac:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

On Linux:
```bash
hostname -I
```

On Windows:
```bash
ipconfig
```

Look for something like `192.168.1.100` (your local network IP)

**3. Run the server:**
```bash
python3 server_mac.py
```

You should see:
```
SERVER READY!
📡 Waiting for Raspberry Pi to connect...
💡 On your Raspberry Pi, use this IP address: 192.168.x.x
```

**Keep this terminal open and note the IP address!**

---

### Part 2: Setup Your Raspberry Pi (Client)

**1. Install dependencies on Raspberry Pi:**
```bash
pip3 install opencv-python numpy pyttsx3
```

**Note:** You do NOT need to install `ultralytics` or `torch` on the Pi! That's the whole point - the heavy lifting happens on your Mac.

**2. If you need espeak for voice alerts (Raspberry Pi OS):**
```bash
sudo apt-get install espeak
```

**3. Run the client (replace with YOUR Mac's IP):**
```bash
python3 pi_client.py --server 192.168.1.100
```

Replace `192.168.1.100` with the IP address shown by the server.

---

## Testing the Connection

**On your Mac terminal, you should see:**
```
✓ Connected to Raspberry Pi at 192.168.x.x
📊 Processed 30 frames | FPS: 25.3 | Detection time: 35.2ms
```

**On your Raspberry Pi, you should see:**
```
✓ Connected to server
DETECTION ACTIVE - Press 'q' to quit
📊 FPS: 24.8 | Utensils: 2 | Hands: 1 | State: ATTENDED
```

---

## Command Line Options

### Client (Raspberry Pi)

```bash
python3 pi_client.py --server IP [OPTIONS]

Required:
  --server IP          Your Mac's IP address (required)

Optional:
  --port PORT          Server port (default: 8888)
  --camera INDEX       Camera index (default: 0)
  --width WIDTH        Frame width (default: 640)
  --height HEIGHT      Frame height (default: 480)
```

**Examples:**

Basic:
```bash
python3 pi_client.py --server 192.168.1.100
```

With custom resolution:
```bash
python3 pi_client.py --server 192.168.1.100 --width 320 --height 240
```

Lower resolution for faster network transfer:
```bash
python3 pi_client.py --server 192.168.1.100 --width 416 --height 416
```

Different camera:
```bash
python3 pi_client.py --server 192.168.1.100 --camera 1
```

---

## Performance Tuning

### If video is laggy or slow:

**1. Reduce resolution (faster network transfer):**
```bash
python3 pi_client.py --server 192.168.1.100 --width 416 --height 416
```

**2. Use wired Ethernet instead of WiFi:**
- Plug both devices into your router with Ethernet cables
- Much faster and more stable than WiFi

**3. Make sure both devices are on the same network:**
- Same WiFi network or same router
- Not using VPN or guest networks

**4. Check network speed:**
On Raspberry Pi:
```bash
ping 192.168.1.100
```
Should show <10ms response time

---

## Troubleshooting

### "Connection refused" error

**Problem:** Server isn't running or firewall is blocking
**Solution:**
1. Make sure server is running on Mac first
2. On Mac, allow Python through firewall:
   - System Settings → Privacy & Security → Firewall → Options
   - Add Python and allow incoming connections
3. Try disabling firewall temporarily to test

### "Connection timeout" error

**Problem:** Wrong IP address or different networks
**Solution:**
1. Double-check Mac's IP address: `ifconfig | grep "inet "`
2. Make sure Pi and Mac are on SAME WiFi network
3. Try pinging from Pi: `ping YOUR_MAC_IP`

### Video shows but no detections

**Problem:** Models not loaded on server
**Solution:**
1. Check server terminal for errors
2. Make sure models downloaded: `yolov8n.pt` and `yolov8n-pose.pt`
3. Restart server

### Voice alerts not working on Pi

**Problem:** TTS not installed
**Solution:**
```bash
sudo apt-get install espeak
pip3 install pyttsx3
```

### Low FPS / Laggy video

**Problem:** Network too slow or resolution too high
**Solution:**
1. Use Ethernet instead of WiFi
2. Reduce resolution: `--width 320 --height 240`
3. Move Pi closer to WiFi router
4. Close other network-heavy apps

### "Camera feed too dark" on Pi

**Problem:** Lighting or camera settings
**Solution:**
The client already has brightness adjustments built-in. If still too dark, edit `pi_client.py` and increase:
```python
frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)  # Increase beta
```

---

## Network Requirements

**Minimum:**
- 5 Mbps network speed
- Same local network (WiFi or Ethernet)
- Port 8888 open (or change to different port)

**Recommended:**
- 10+ Mbps network speed
- Wired Ethernet connection
- Low latency (<20ms ping time)

---

## Expected Performance

**Typical FPS:**
- WiFi (good signal): 15-25 FPS
- Ethernet: 25-30 FPS
- WiFi (weak signal): 5-15 FPS

**Detection latency:**
- Network delay: 10-50ms
- Detection time: 30-50ms
- Total: ~40-100ms per frame

---

## Stopping the System

**On Raspberry Pi:**
- Press 'q' in the video window, or Ctrl+C in terminal

**On Mac:**
- Press Ctrl+C in terminal
- Server will wait for new connections

---

## Advanced: Running Server in Background

**On Mac (keep server running even after closing terminal):**
```bash
nohup python3 server_mac.py > server.log 2>&1 &
```

Check if running:
```bash
ps aux | grep server_mac
```

Stop background server:
```bash
pkill -f server_mac.py
```

---

## Security Notes

⚠️ This setup is designed for **local network use only**
- Server listens on all interfaces (0.0.0.0)
- No authentication or encryption
- **Do not expose to the internet**
- Use only on trusted local networks (home/office WiFi)

For internet access, you would need to add:
- Authentication
- Encryption (SSL/TLS)
- Firewall rules
- Port forwarding configuration

---

## Summary

1. Run server on Mac: `python3 server_mac.py`
2. Note the IP address shown
3. Run client on Pi: `python3 pi_client.py --server YOUR_MAC_IP`
4. Both should connect and show detection!

**Benefits of this setup:**
- ✅ Real-time detection (20-30 FPS)
- ✅ No expensive hardware needed
- ✅ Can use GPU on Mac for faster processing
- ✅ All features work: detection, tracking, voice alerts
- ✅ Easy to update models (just update on server)
