# Computer Vision Measurement System

A real-time, interactive computer vision measurement tool built with OpenCV.
This application enables **precise distance and angle measurements** using a webcam or static images, enhanced by a **modern glassmorphic UI**, zoom/pan controls, and intelligent calibration options.

---

## ✨ Features

### 🎯 Measurement Capabilities
- **Distance Measurement**
  - Pixel-based
  - Real-world (cm) using depth or reference calibration
- **Angle Measurement**
  - Three-point angle calculation
  - Vertex-first workflow with visual arc rendering
- **Hover & Delete**
  - Hover to highlight measurements
  - Right-click to delete any measurement

### 📷 Input Modes
- **Live Webcam Feed**
- **Static Image Measurement**
  - Load images directly via path input
  - Automatic resizing for performance

### 🎛 Calibration Options
- **Default Camera Calibration**
- **Reference-Based Calibration**
  - Define real-world size for accurate measurements
- **Depth-Based Estimation**
  - Adjustable depth (I / O keys)

### 🔍 View Controls
- **Zoom In / Out** (up to 5×)
- **Pan** (Left / Right / Up / Down)
- Accurate coordinate mapping between zoomed and original frames

### 🖥 UI & UX
- Modern **glassmorphic interface**
- Dynamic buttons and hover effects
- Measurement labels with smart positioning
- Keyboard shortcut overlay
- Visual LIVE indicator

---

## 🧰 Technology Stack

- **Python 3.12(required version)**
- **OpenCV**
- **NumPy**
- Native OpenCV GUI (no external UI frameworks)

**Notes**
we also use `pickle`, `os`, `pathlib`, and `math`-related utilities which are part of Python’s standard library.

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/advanced-cv-measurement.git
cd advanced-cv-measurement
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

mentionned in requirements.txt

---

## ▶️ Running the Application

```bash
python measurement_app.py
```

Make sure:

* A webcam is connected (for live mode)
* OpenCV can access your camera

---

## ⌨ Keyboard Shortcuts

| Key           | Action                       |
| ------------- | ---------------------------- |
| Q             | Quit application             |
| D             | Distance mode                |
| A             | Angle mode                   |
| C             | Calibration mode             |
| X             | Reset calibration            |
| L             | Load image                   |
| V             | Switch to live video         |
| R             | Reset all measurements       |
| BACKSPACE     | Remove last point            |
| I / O         | Increase / decrease depth    |
| + / -         | Zoom in / out                |
| 1 / 2 / 3 / 4 | Pan left / right / up / down |
| H             | Toggle shortcuts panel       |
| Right Click   | Delete measurement           |

---

## 🖱 Mouse Controls

* **Left Click**

  * Add measurement points
* **Right Click**

  * Delete existing measurement
* **Hover**

  * Highlight measurements

---

## 📐 Measurement Workflow

### Distance

1. Select **Distance Mode**
2. Click **two points**
3. View pixel or real-world distance

### Angle

1. Select **Angle Mode**
2. Click **vertex first**
3. Click two arm points
4. View angle in degrees

### Calibration

1. Select **Calibration Mode**
2. Click two points on a known object
3. Enter real-world distance (cm)

---


## ⚠ Limitations & Notes

* Real-world measurements depend on accurate calibration.
* Depth-based estimation is an approximation.
* Designed primarily for **single-camera, planar measurements**.

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## ⭐ Contributing

This is a learning project purely made for entertainment purposes, if you have any fun suggestions or ideas dont hesitate to contact me on linked in or via email <3.
Happy coding!

---
