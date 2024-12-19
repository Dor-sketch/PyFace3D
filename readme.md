<div align="center">

# PyFace3D - Face Model Generator

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Dor-sketch/PyFace3D/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Dor-sketch/PyFace3D.svg)](https://GitHub.com/Dor-sketch/PyFace3D/issues/)
[![Documentation Status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://Dor-sketch.github.io/PyFace3D/)

<img src="docs/og-image.svg" alt="PyFace3D Banner" width="800"/>

Real-time 3D face modeling application that captures facial landmarks using your webcam and generates a textured 3D mesh model. Features split-screen interface showing both live camera feed and interactive 3D model.

[Demo](https://dorpascal.com/PyFace3D/) •
[Installation](#installation) •
[Documentation](#technical-details) •
[Contributing](#contributing)

</div>

---

## ✨ Features

- 🎥 Real-time face landmark detection using MediaPipe
- 🔮 Live 3D mesh generation with Delaunay triangulation
- 🖼️ Interactive 3D model viewer with texture mapping
- 📺 Split-screen display showing camera feed and 3D model
- 🖱️ Model rotation using mouse controls
- 💾 Export functionality for 3D models (OBJ format with textures)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- OpenGL-compatible graphics card
- Webcam
- 4GB+ RAM

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Dor-sketch/PyFace3D.git
    cd PyFace3D
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0%2B-red.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24.0%2B-blue.svg)](https://numpy.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0%2B-green.svg)](https://mediapipe.dev/)
[![PyGame](https://img.shields.io/badge/PyGame-2.5.0%2B-yellow.svg)](https://www.pygame.org/)
[![PyOpenGL](https://img.shields.io/badge/PyOpenGL-3.1.7%2B-orange.svg)](http://pyopengl.sourceforge.net/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11.0%2B-purple.svg)](https://scipy.org/)

## 🎮 Usage

1. Activate virtual environment (if used):

    ```bash
    # Windows
    .\venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate
    ```

2. Run application:

    ```bash
    python3 main.py
    ```

### Controls

| Key/Action | Description |
|------------|-------------|
| Left Mouse + Drag | Rotate 3D model |
| S | Save model as OBJ |
| R | Reset rotation |
| Close Window | Exit application |

## 🏗️ Architecture

<details>
<summary>Click to expand architecture details</summary>

### 1. Face Detection and Landmark Tracking

- MediaPipe Face Mesh (468 landmarks)
- Real-time confidence scoring
- Smooth landmark transitions

### 2. 3D Mesh Generation

- 3D coordinate mapping
- Delaunay triangulation
- Normal vector calculation
- Texture coordinate mapping

### 3. Rendering Pipeline

- OpenGL-based rendering
- Split viewport management
- Dynamic texture updating
- Lighting system

### 4. Performance Optimization

- Configurable frame buffering
- Resolution scaling
- Memory management
- Vertex optimization

</details>

## 💾 Output Files

Models are saved with timestamp-based naming:

```plaintext
face_model_[timestamp].obj    # Geometry
face_model_[timestamp].mtl    # Materials
face_texture_[timestamp].png  # Texture
```

## 🔧 Troubleshooting

This project has been tested on a MacBook Air M2 using Python 3.11.

<details>
<summary>Common Issues</summary>

### Camera Detection

- ✓ Check webcam connection
- ✓ Verify system permissions
- ✓ Try alternate camera index

### Performance

- ✓ Adjust process_resolution
- ✓ Modify max_buffer_size
- ✓ Ensure proper lighting

### OpenGL

- ✓ Update graphics drivers
- ✓ Verify PyOpenGL installation
- ✓ Check version compatibility

</details>

## 🤝 Contributing

[![Contributors](https://img.shields.io/github/contributors/Dor-sketch/PyFace3D.svg)](https://github.com/Dor-sketch/PyFace3D/graphs/contributors)

Contributions are welcome! Contributions guidelines will be added soon.

## 📝 License

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for face mesh solution
- OpenGL and PyOpenGL communities
- Contributors to numpy, scipy, and pygame

---

<div align="center">

Made with ❤️ by [Dor Pascal](https://dorpascal.com/)

[![GitHub](https://img.shields.io/badge/GitHub-Dor--sketch-lightgrey.svg?logo=github)](https://github.com/Dor-sketch)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Dor%20Pascal-blue.svg?logo=linkedin)](https://www.linkedin.com/in/dor-pascal/)
[![Twitter](https://img.shields.io/badge/Twitter-DorPascalLab-blue.svg?logo=twitter)](https://twitter.com/DorPascalLab)

[![Star History](https://img.shields.io/github/stars/Dor-sketch/PyFace3D.svg?style=social)](https://github.com/Dor-sketch/PyFace3D/stargazers)
[![Follow](https://img.shields.io/github/followers/Dor-sketch.svg?style=social&label=Follow)](https://github.com/Dor-sketch)

</div>
