# Driver Drowsiness Detection System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-Latest-red.svg)](http://dlib.net/)

A real-time driver drowsiness detection system using computer vision and facial landmark detection to enhance road safety.

## Project Structure
```
driver-drowsiness-detection/
├── Driver-Drowsiness-Detection.py    # Main detection script
├── ref_notebook.ipynb                # Reference notebook with development process
├── shape_predictor_68_face_landmarks.dat  # Facial landmark predictor model
├── eye1.jpg                          # Sample test image
├── facial_landmarks.png              # Visualization of facial landmarks
└── README.md
```

## Features

- Real-time facial landmark detection
- Eye aspect ratio (EAR) calculation
- Drowsiness alert system
- Visual feedback system
- Frame-by-frame analysis

## Technical Details

### Core Components

1. **Facial Detection**
   - Uses dlib's frontal face detector
   - Robust against varying lighting conditions
   - Real-time processing capability

2. **Landmark Detection**
   - 68-point facial landmark detection
   - Specific focus on eye region landmarks
   - High-precision coordinate mapping

3. **Drowsiness Detection**
   - Eye Aspect Ratio (EAR) calculation
   - Temporal analysis of eye closure
   - Configurable alert thresholds

## Prerequisites

- Python 3.8+
- OpenCV
- dlib
- numpy
- imutils
- scipy
- pygame (for audio alerts)

## How It Works

1. **Face Detection**
   - Captures video feed from webcam
   - Detects faces in each frame
   - Extracts face region for processing

2. **Landmark Detection**
   - Maps 68 facial landmarks
   - Identifies eye regions
   - Calculates eye aspect ratio

3. **Drowsiness Analysis**
   - Monitors eye aspect ratio over time
   - Detects prolonged eye closure
   - Triggers alerts when drowsiness detected

## Alert System

The system provides multiple types of alerts:
- Visual warnings on screen
- Audio alerts for immediate attention
- Frame counting for drowsiness duration

## Limitations

- Requires good lighting conditions
- May be affected by wearing glasses
- Needs consistent face visibility
- Processing speed depends on hardware

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- dlib development team
- OpenCV community
- Contributors and testers
