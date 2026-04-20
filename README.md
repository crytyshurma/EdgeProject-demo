# 🎥 Jetson Nano Multi-Camera Surveillance System

A real-time multi-camera surveillance system optimized for **Jetson Nano**, supporting:

* 📷 Multiple camera inputs (USB + video files)
* 🧠 Object detection (PyTorch models)
* 🧍 Tracking (lightweight SORT / IoU)
* 🚨 Alert system with snapshots
* 🎬 Grid-based video recording (FFmpeg)

---

# ⚙️ System Architecture

```
Cameras → Detection → Tracking → Alerts → Recorder (Grid Video)
```

* **Detection**: PyTorch (FasterRCNN / SSD)
* **Tracking**: Lightweight tracker (SORT / IoU)
* **Recorder**: FFmpeg via rawvideo pipe
* **Execution**: Sequential pipeline

---

# 📦 Requirements

## Hardware

* Jetson Nano (JetPack 4.x)
* USB Cameras OR video files

## Software

* Docker with NVIDIA runtime
* Python 3.6+

---

# 🚀 Setup & Run

## 1. Run Docker Container

```bash
sudo docker run -it \
--name jetson-surveillance \
--runtime nvidia \
--network host \
--ipc host \
--privileged \
--device /dev/video0 \
--device /dev/video1 \
--device /dev/video2 \
--device /dev/video3 \
-v /sdcard/fol:/workspace \
nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
```

👉 To allow all devices:

```bash
--device /dev:/dev
```

---

## 2. Install Dependencies

### System packages

```bash
apt update
apt install -y ffmpeg vim python3-opencv python3-numpy
```

### Python packages

```bash
pip3 install python-dotenv loguru Pillow numpy
```

---

## 3. Project Structure

```
magneton-demo/
│
├── core/
│   ├── detector.py
│   ├── tracker.py
│   ├── recorder.py
│   └── alert.py
│
├── utils/
│   ├── drawing.py
│   ├── helpers.py
│   └── logger.py
│
├── data/
│   ├── cam1.mp4
│   ├── cam2.mp4
│   ├── cam3.mp4
│   ├── recordings/
│   └── snapshots/
│
├── config.py
└── main.py
```

---

## 4. Configure Cameras

Edit `config.py`:

```python
CAMERA_SOURCES = [
    0,
    "data/cam1.mp4",
    "data/cam2.mp4",
    1
]
```

---

## 5. Run the Project

```bash
cd /workspace
python3 main.py
```

---

# 📊 Output

## 🎬 Recordings

```
data/recordings/
```

## 📸 Snapshots

```
data/snapshots/
```

---

# 🧠 Key Features

* Multi-camera grid recording
* Real-time detection + tracking
* Alert system with cooldown
* Works headless
* Jetson optimized

---

# ⚡ Performance Notes

* Sequential processing
* ~1–2 FPS per camera (4 cameras)
* Detector is bottleneck

---

# 🚀 Future Improvements

* Multi-threaded detection
* TensorRT optimization
* Async pipelines
* RTSP streaming

---

# 🐳 Docker Tips

### Save container state

```bash
docker commit jetson-surveillance jetson-surveillance:latest
```

### Run saved image

```bash
docker run jetson-surveillance:latest
```

---

# 🛠 Troubleshooting

## No detections

```python
CONFIDENCE_THRESH = 0.2
```

## Camera not detected

```bash
ls /dev/video*
```

## Corrupted video

Stop using:

```bash
CTRL + C
```

---

# 📌 Summary

* Multi-camera edge AI system
* Detection + tracking + recording
* Jetson Nano optimized
* Modular and extendable
