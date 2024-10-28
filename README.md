
# Driver Behavior Monitoring System 🚗💤

This project is a **real-time driver monitoring system** designed to detect signs of **drowsiness, yawning, and distraction** using computer vision and audio alerts. Built with **OpenCV**, **MediaPipe**, and **Pygame**, it aims to improve road safety by alerting drivers when signs of fatigue or inattention are detected.

## Key Features
- **Face Detection & Landmark Detection**: Utilizes MediaPipe’s face detection and face mesh to detect facial landmarks in real time.
- **Behavior Analysis**:
  - **Yawning Detection**: Measures mouth openness to detect yawns.
  - **Eye Closure Detection**: Monitors eye openness to identify signs of drowsiness.
  - **Head Pose Analysis**: Estimates head position to detect when the driver is looking away from the road.
- **Audio Alerts**: Plays specific alert sounds (yawn, drowsiness, and distraction) when risky behaviors are detected, using Pygame for sound management.
- **Screenshot Capture**: Automatically captures and saves screenshots in designated folders when signs of dangerous behaviors are identified for future analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/driver-behavior-monitoring.git
   cd driver-behavior-monitoring
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have **OpenCV**, **MediaPipe**, and **Pygame** installed, and have the alert sound files in the project directory.

## Usage
Run the code with:
```bash
python driver_behavior_monitor.py
```
Use `q` to quit the application.

## Future Improvements
- **Customizable Detection Sensitivity**: Fine-tune detection thresholds for personalized monitoring.
- **Performance Optimization**: Enhance frame processing for higher FPS on lower-end hardware.

## Notes
This is a prototype for educational purposes. It’s advised to test in a controlled environment before real-world usage.
