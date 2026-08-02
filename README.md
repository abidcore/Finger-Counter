<p align="center">
  <img src="assets/logo.png" alt="AI Finger Counter System logo" width="700">
</p>

<h1 align="center">AI Finger Counter System</h1>

<p align="center">
  A real-time, production-grade computer vision application that detects hands, tracks 21 landmarks per hand, counts extended fingers (0–5), and distinguishes left from right hands — built with a clean, modular, testable software architecture.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-orange">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Stable-brightgreen">
</p>

---

## 📌 Project Description

**AI Finger Counter System** is a real-time hand-tracking application that uses your webcam to detect one or two hands, map 21 3D landmarks per hand with **MediaPipe**, and geometrically determine how many fingers are extended — from a closed fist (0) to an open palm (5). Unlike typical single-script tutorial projects, this system is engineered as a **modular Python package** with clear separation of concerns: detection, counting logic, gesture classification, temporal stabilization, FPS profiling, and UI rendering each live in their own dedicated module.

The result is a responsive, flicker-free, professional on-screen overlay that simultaneously tracks **both hands independently**, labels them as **Left**/**Right**, displays a **live finger count**, a **named gesture** (e.g. *Peace*, *Thumbs Up*, *Open Palm*), a **detection confidence bar**, a smoothed **FPS counter**, and a **webcam connectivity indicator** — all while gracefully handling camera failures and unexpected runtime errors.

This project was built as part of an **AI & Machine Learning Diploma** portfolio to demonstrate applied computer vision engineering, clean architecture, and professional documentation practices suitable for GitHub, LinkedIn, and academic evaluation.

---

## ✨ Features

| Category | Capability |
|---|---|
| **Detection** | Real-time hand detection and 21-point landmark tracking via MediaPipe |
| **Counting** | Accurate finger counting (0–5) using dual geometric heuristics (vertical joint comparison + normalized thumb-distance method) |
| **Multi-Hand** | Simultaneous, independent tracking of **both** left and right hands |
| **Gesture Intelligence** | Named gesture classification (Fist, Peace, Thumbs Up, Open Palm, Pointing, etc.) |
| **Stabilization** | Sliding-window majority-vote smoothing eliminates frame-to-frame count flicker |
| **Performance** | Exponentially-smoothed, real-time FPS counter |
| **Confidence** | Live handedness detection-confidence percentage and visual bar |
| **Reliability** | Webcam connectivity status indicator with automatic reconnect attempts |
| **Robustness** | Full exception handling — camera-unavailable errors are caught and reported cleanly instead of crashing |
| **UX** | Keyboard shortcuts: `Q` / `ESC` to exit, `S` to save a timestamped snapshot |
| **UI/UX** | Professional translucent overlay panels, rounded corners, per-hand bounding boxes, drop-shadow text |
| **Engineering** | Modular OOP architecture, PEP8-compliant, fully typed, docstring-documented, unit-testable logic layer |

---

## 🛠️ Technology Stack

- **Language:** Python 3.12+
- **Computer Vision:** [OpenCV](https://opencv.org/) — video capture, image processing, UI rendering
- **Hand Tracking Model:** [MediaPipe Hands](https://developers.google.com/mediapipe) — 21-point landmark detection
- **Numerical Computing:** [NumPy](https://numpy.org/) — array and image-buffer operations
- **Architecture:** Modular, object-oriented, single-responsibility Python package

---

## 📥 Installation

### Prerequisites
- Python **3.12 or higher**
- A working webcam
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/abidcore/AI-Finger-Counter.git
cd AI-Finger-Counter

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application from the project root:

```bash
python main.py
```

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` or `ESC` | Exit the application |
| `S` | Save a timestamped snapshot to `snapshots/` |

> 💡 **Tip:** Hold your hand fully inside the camera frame with good lighting for the most accurate detection confidence scores.

### Running in VS Code
1. Open the project folder in VS Code.
2. Select the interpreter from your `venv` (`Ctrl+Shift+P` → *Python: Select Interpreter*).
3. Run `main.py` directly, or use the integrated terminal: `python main.py`.

---

## 📁 Folder Structure

```
AI-Finger-Counter/
│
├── main.py                  # Application entry point & orchestration
├── requirements.txt         # Python dependencies
├── README.md                 # Project documentation (this file)
├── LICENSE                   # MIT License
├── .gitignore                 # Git ignore rules
│
├── src/                       # Core application package
│   ├── __init__.py            # Package exports
│   ├── hand_tracker.py        # MediaPipe Hands wrapper & landmark extraction
│   ├── finger_counter.py      # Geometric finger-counting logic
│   ├── gesture_detector.py    # Stabilization + named gesture classification
│   ├── fps.py                 # Smoothed FPS counter
│   └── utils.py                # Reusable OpenCV drawing/UI helpers
│
├── config/                    # Centralized configuration
│   ├── __init__.py
│   └── settings.py             # All tunable constants (camera, colors, thresholds, UI)
│
├── assets/                    # Visual assets
│   ├── logo.png
│   └── demo.png
│
└── docs/                       # Extended documentation
    └── project_report.md       # Full academic-style project report
```

---

## 🖼️ Screenshots

<p align="center">
  <img src="assets/demo.png" alt="Application UI demo" width="750">
</p>

> The image above is an illustrative UI mockup generated to preview the layout. Replace `assets/demo.png` with an actual screenshot or GIF of the running application for your submission/portfolio.

---

## ✅ Advantages

- **Modular & maintainable** — each concern (tracking, counting, UI, stabilization) is isolated in its own testable module.
- **Orientation-independent thumb logic** — uses a normalized distance heuristic instead of fragile left/right branching.
- **Flicker-free output** — sliding-window majority voting produces a stable, professional-looking count.
- **Dual-hand aware** — tracks and labels both hands independently and simultaneously.
- **Fails gracefully** — camera disconnection and unexpected exceptions are caught and reported, not crashed.
- **Configurable** — every threshold, color, and dimension lives in one settings file.
- **Portfolio-ready** — clean PEP8 code, full docstrings, and academic-grade documentation.

---

## 🚀 Future Scope

- Add support for **custom gesture-to-action mapping** (e.g. volume control, slide navigation, virtual mouse).
- Integrate a **TensorFlow/PyTorch classifier** for dynamic gesture sequences (not just static counts).
- Add a **REST/WebSocket API** to stream finger-count data to external applications.
- Build a **Streamlit/Flask web dashboard** for browser-based access without a local Python environment.
- Add **unit test suite** (pytest) covering `finger_counter.py` and `gesture_detector.py` with CI integration.
- Support **sign-language digit recognition** as an extension of the counting logic.
- Add **multi-camera / IP-camera** input support.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abid Ali**
AI & Machine Learning Diploma Student

- GitHub: [github.com/abidcore](https://github.com/abidcore)
- LinkedIn: [linkedin.com/in/abid-ali-shaikh-03a591423](https://www.linkedin.com/in/abid-ali-shaikh-03a591423)
- Email: [abidalishaikh2007@gmail.com](mailto:abidalishaikh2007@gmail.com)

---

<p align="center">⭐ If you found this project useful, consider giving it a star on GitHub!</p>
