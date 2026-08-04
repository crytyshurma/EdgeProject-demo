# 🚀 Jetson Nano Multi-Camera Edge AI Surveillance System

A real-time **Edge AI surveillance system** optimized for the **NVIDIA Jetson Nano**. The system processes multiple camera streams simultaneously, performs **object detection**, **multi-object tracking**, **alert generation**, **snapshot capture**, and **continuous grid-based video recording** without relying on cloud processing.

---

# 🎯 Objective

Develop a lightweight, real-time surveillance system capable of monitoring multiple camera feeds on an edge device by performing:

- 🎯 Real-time object detection
- 👥 Multi-object tracking
- 🚨 Alert generation for newly detected objects
- 📸 Automatic snapshot capture
- 🎥 Continuous surveillance video recording
- ⚡ Low-latency on-device inference

---

# ✨ Key Features

- 📷 Supports multiple USB cameras and video files
- 🧠 Real-time object detection using **Faster R-CNN MobileNetV3 FPN**
- 👤 Lightweight IoU-based multi-object tracking
- 🆔 Persistent object IDs across consecutive frames
- 🚨 Alert generation whenever a new object enters the scene
- 📸 Automatic snapshot saving during alerts
- 🎥 Continuous grid-based MP4 recording using FFmpeg
- 🐳 Dockerized deployment for easy portability
- ⚡ Optimized for NVIDIA Jetson Nano

---

# ⚙️ System Architecture

```text
              Multiple Cameras
                     │
                     ▼
          Frame Capture (OpenCV)
                     │
                     ▼
              Frame Preprocessing
                     │
                     ▼
      Faster R-CNN Object Detection
                     │
                     ▼
          IoU-based Object Tracking
          │                       │
          │                       ▼
          │               Alert Manager
          │                       │
          │               Snapshot Capture
          │
          ▼
Draw Bounding Boxes & Tracking IDs
                     │
                     ▼
          Multi-Camera Grid Builder
                     │
                     ▼
             FFmpeg Video Encoder
                     │
                     ▼
          Single Surveillance MP4
```

---

# 🧠 Detection Module

- 🔹 Framework: PyTorch (Torchvision)
- 🔹 Model: Faster R-CNN MobileNetV3 Large 320 FPN
- 🔹 Purpose:
  - Detect objects in every frame
  - Generate bounding boxes and confidence scores

---

# 👤 Tracking Module

- 🔹 Algorithm: Custom IoU-based Tracker
- 🔹 Assigns persistent IDs to detected objects
- 🔹 Matches detections using Intersection over Union (IoU)
- 🔹 Generates a new ID whenever a new object appears

---

# 🚨 Alert System

Alerts are generated whenever:

- ✅ A new tracking ID is created
- ✅ A new object enters the camera view

Each alert triggers:

- 📸 Snapshot capture
- 📝 Alert logging
- ⏳ Cooldown mechanism to avoid repeated alerts

---

# 🎥 Recording Pipeline

- 📹 Continuously records all processed frames
- 🖥️ Combines all camera feeds into a single surveillance grid
- 🎞️ Uses FFmpeg to encode frames into H.264 MP4 format
- 💾 Stores recordings locally on the Jetson Nano

---

# 📂 Output

## 🎥 Recordings

```
data/recordings/
```

Example

```
surveillance_20260804_103000.mp4
```

---

## 📸 Snapshots

```
data/snapshots/
```

Example

```
cam0_20260804_103015.jpg
```

---

# 📁 Project Structure

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
│   ├── recordings/
│   ├── snapshots/
│   └── sample_videos/
│
├── config.py
└── main.py
```

---

# 🛠️ Technologies Used

### 💻 Programming

- 🐍 Python

### 🤖 AI & Computer Vision

- 🧠 PyTorch
- 👁️ Torchvision
- 📷 OpenCV
- 🎯 Faster R-CNN MobileNetV3 FPN

### 🎥 Multimedia

- 🎬 FFmpeg (H.264 Video Encoding)

### ⚙️ Deployment

- 🐳 Docker
- 🚀 NVIDIA Jetson Nano

---

# 📊 Performance

- ⚡ Sequential processing pipeline
- 📷 Supports multiple simultaneous camera streams
- 🎥 Continuous surveillance recording
- 🖥️ Approximately **1–2 FPS per camera (4 cameras)**
- 🚧 Object detection is the primary performance bottleneck

---

# 🚀 Future Improvements

- ⚡ TensorRT optimization
- 🧵 Multi-threaded detection pipeline
- 👤 Deep SORT / ByteTrack integration
- 📡 RTSP camera support
- 📧 Email/SMS alert notifications
- ☁️ Optional cloud dashboard integration
- 🎥 Hardware-accelerated video encoding

---

# 📌 Summary

✅ Multi-camera Edge AI surveillance system

✅ Real-time object detection using Faster R-CNN

✅ IoU-based multi-object tracking

✅ Continuous grid-based MP4 recording

✅ Automatic alert generation

✅ Snapshot capture on alerts

✅ Dockerized deployment

✅ Optimized for NVIDIA Jetson Nano
