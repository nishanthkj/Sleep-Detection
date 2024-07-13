# Eye Closure Detection System

## Overview

This project is a real-time eye closure detection system that uses a webcam feed to monitor eye closures. It calculates the Eye Aspect Ratio (EAR) to determine if the eyes are closed and triggers an alarm if they remain closed for a specified number of consecutive frames. This can be useful for detecting drowsiness in drivers, ensuring alertness in security personnel, and various other applications.

## Project Structure

```
Sleep_Detection/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat
│
├── eye_closure_detection.py
├── requirements.txt
└── README.md
```

## Files Description

- **models/shape_predictor_68_face_landmarks.dat**: The pre-trained dlib shape predictor model file.
- **eye_closure_detection.py**: The main Python script for real-time eye closure detection.
- **requirements.txt**: List of dependencies required for the project.
- **README.md**: Documentation for the project.

## Setup Instructions

### Prerequisites

- Python 3.12

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/nishanthkj/Sleep_Detection
    cd Sleep_Detection
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the shape predictor model**:
    - Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the `models` directory.

## Usage

1. **Run the eye closure detection script**:
    ```bash
    python eye_closure_detection.py
    ```

2. **Allow webcam access**:
    - The script will access your webcam. Ensure you have a working webcam connected.

3. **Monitor the output**:
    - The script will display the webcam feed and highlight detected faces and eye regions. It will trigger an alarm if the eyes remain closed for the specified number of consecutive frames.

## Code Explanation

### `eye_closure_detection.py`

This is the main Python script for real-time eye closure detection.

```python
"""
This module detects eye closure in real-time using a webcam feed. It calculates the
Eye Aspect Ratio (EAR) to determine if the eyes are closed and triggers an alarm if
they remain closed for a specified number of consecutive frames. This can be useful
for detecting drowsiness in drivers, ensuring alertness in security personnel, and
various other applications.
"""

import winsound  # Standard import should be first
import cv2  # pylint: disable=import-error
import dlib  # pylint: disable=import-error
import numpy as np
from scipy.spatial import distance as dist  # pylint: disable=import-error

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.
    
    Parameters:
    eye (numpy.ndarray): Array of coordinates for the eye landmarks.
    
    Returns:
    float: The calculated EAR value.
    """
    dist_a = dist.euclidean(eye[1], eye[5])
    dist_b = dist.euclidean(eye[2], eye[4])
    dist_c = dist.euclidean(eye[0], eye[3])
    aspect_ratio = (dist_a + dist_b) / (2.0 * dist_c)
    return aspect_ratio

# Constants for EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Initialize the frame counters and the alarm status
COUNTER = 0
ALARM_ON = False

# Load the pre-trained face detector and shape predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # pylint: disable=no-member
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Grab the indexes of the facial landmarks for the left and right eye
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# Start the video stream
cap = cv2.VideoCapture(0)  # pylint: disable=no-member

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

        # Ensure the grayscale image is 8-bit
        if gray.dtype != np.uint8:
            gray = gray.astype('uint8')

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  # pylint: disable=no-member
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        winsound.Beep(2500, 1000)  # Beep at 2500 Hz for 1 second
            else:
                COUNTER = 0
                ALARM_ON = False

            # Draw the face bounding box and eye regions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # pylint: disable=no-member
            for (ex, ey) in np.concatenate((left_eye, right_eye), axis=0):
                cv2.circle(frame, (ex, ey), 2, (0, 255, 0), -1)  # pylint: disable-no-member

        cv2.imshow("Frame", frame)  # pylint: disable=no-member
        if cv2.waitKey(1) & 0xFF == ord('q'):  # pylint: disable-no-member
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()  # pylint: disable-no-member
```

### `requirements.txt`

This file lists all the dependencies required for the project.

```
cmake==3.30.0
dlib @ file:///D:/Projects/Python/Sleep%20Detection/dlib-19.24.99-cp312-cp312-win_amd64.whl#sha256=20c62e606ca4c9961305f7be3d03990380d3e6c17f8d27798996e97a73271862
numpy==1.26.4
opencv-python==4.10.0.84
PyOpenGL==3.1.7
scipy==1.14.0
```

## Conclusion

This documentation provides a comprehensive guide to setting up and running the Eye Closure Detection System. By following the steps outlined, you should be able to deploy the application and monitor eye closures in real-time. If you encounter any issues, ensure that all dependencies are installed and that the model file is correctly placed in the `models` directory.

Feel free to customize and enhance the functionality as per your requirements. This project serves as a foundational implementation for real-time eye closure detection.