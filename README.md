# Face Expression Detection 😊😔

A real-time facial expression detection program that uses your webcam to detect whether you're happy or sad based on smile detection. Built with Python and OpenCV.

## Features

- **Real-time Face Detection**: Automatically detects faces using Haar Cascade classifiers
- **Smile Recognition**: Analyzes facial expressions to determine if you're smiling
- **Expression Classification**: Labels your expression as "happy" (😊) or "sad" (😔)
- **Smooth Detection**: Uses a 15-frame sliding window to prevent flickering results
- **User-Friendly Interface**: 
  - Mirror-mode display for natural viewing
  - Color-coded feedback (green for happy, yellow for sad, red for no face)
  - Simple keyboard controls

## Demo

The program displays:
- A bounding box around detected faces
- Your current expression (happy/sad/no face) with color-coded text
- Real-time video feed from your webcam

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- A working webcam

## Installation

1. Clone this repository:
```bash
git clone https://github.com/JessieLYeung/ComputerVision.git
cd ComputerVision
```

2. Install the required dependencies:
```bash
pip install opencv-python
```

3. Ensure your system allows camera access:
   - **Windows**: Settings → Privacy → Camera
   - **macOS**: System Preferences → Security & Privacy → Camera
   - **Linux**: Check camera permissions for your user

## Usage

Run the program:
```bash
python app.py
```

**Controls:**
- Press **'q'** to quit the application

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in each frame
2. **Region of Interest**: Focuses on the lower half of the face (mouth region) for smile detection
3. **Smile Detection**: Analyzes the mouth region to identify smiles
4. **Smoothing Algorithm**: Averages results over 15 frames to provide stable, flicker-free labels
5. **Classification**: Labels expression as "happy" if ≥50% of recent frames show a smile, otherwise "sad"

## Technical Details

- **Face Detection Model**: `haarcascade_frontalface_default.xml`
- **Smile Detection Model**: `haarcascade_smile.xml`
- **Smoothing Window**: 15 frames (adjustable in code)
- **Video Processing**: Converts frames to grayscale for faster processing
- **Detection Parameters**: Tuned for stability (scaleFactor=1.4, minNeighbors=18 for smiles)

## Code Structure

```
computer vision/
├── app.py          # Main program file
└── README.md       # This file
```

### Key Functions:
- `put_text()`: Helper function to display text on video frames
- `run_face()`: Main function handling face detection, smile recognition, and video display

## Customization

You can adjust detection sensitivity by modifying parameters in `app.py`:

```python
# Face detection sensitivity
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
)

# Smile detection sensitivity
smiles = smile_cascade.detectMultiScale(
    mouth_region, scaleFactor=1.4, minNeighbors=18, minSize=(30, 30)
)

# Smoothing window size
happy_votes = deque(maxlen=15)  # Adjust maxlen for more/less smoothing
```

## Troubleshooting

**Camera not opening:**
- Check camera permissions in your system settings
- Ensure no other application is using the camera
- Try changing the camera index: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

**Poor detection accuracy:**
- Ensure good lighting conditions
- Position your face clearly in front of the camera
- Adjust the `scaleFactor` and `minNeighbors` parameters

**Program runs slowly:**
- Close other resource-intensive applications
- Reduce video resolution (modify capture settings in code)

## Future Enhancements

Possible improvements for this project:
- [ ] Add more emotion categories (surprise, anger, neutral)
- [ ] Track happiness percentage over time
- [ ] Multi-face detection and tracking
- [ ] Save screenshots when happy
- [ ] Add FPS counter
- [ ] Create GUI with adjustable settings
- [ ] Integrate modern ML models (MediaPipe, dlib)

## License

This project is open source and available for educational purposes.

## Author

Created by [Jessie Yeung](https://github.com/JessieLYeung)

## Acknowledgments

- Built with [OpenCV](https://opencv.org/)
- Uses pre-trained Haar Cascade classifiers from the OpenCV library
