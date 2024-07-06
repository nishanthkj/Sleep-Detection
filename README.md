# Eye Closure Detection with Alarm

This project detects eye closure using a webcam and sounds an alarm if the eyes remain closed for a specified duration. The detection is based on the Eye Aspect Ratio (EAR) calculated from facial landmarks.

## Requirements

To run this project, you need to install the following libraries:

- OpenCV
- dlib
- NumPy
- SciPy

## Installation

### Step 1: Create a Virtual Environment

Creating a virtual environment is recommended to manage dependencies and avoid conflicts.

```sh
# Create a virtual environment
python -m venv ear_detection_env

# Activate the virtual environment
# On Windows
ear_detection_env\Scripts\activate

# On macOS/Linux
source ear_detection_env/bin/activate
```

### Step 2: Install Required Libraries

Install the necessary libraries using `pip`.

```sh
# Install OpenCV for computer vision tasks
pip install opencv-python

# Install dlib for machine learning and computer vision tasks
pip install dlib

# Install NumPy for numerical operations
pip install numpy

# Install SciPy for spatial distance calculations
pip install scipy
```

### Step 3: Download the Shape Predictor Model

Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Extract the file if it is compressed (e.g., `.bz2`), and place the `.dat` file in the same directory as your script or provide the correct path to the file.

## Script

Save the following script as `eye_closure_detection.py`:

```python
import cv2  # OpenCV library for computer vision tasks
import dlib  # Dlib library for machine learning and computer vision tasks
import numpy as np  # NumPy library for numerical operations
from scipy.spatial import distance as dist  # Scipy library for spatial distance calculations
import winsound  # For alarm sound on Windows

def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR).
    """
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Constants for EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Initialize the frame counters and the alarm status
COUNTER = 0
ALARM_ON = False

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()  # Initialize dlib's face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Update this path if necessary

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Start the video stream
cap = cv2.VideoCapture(0)  # Open the default camera

try:
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret or frame is None:
            print("Failed to capture frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

        # Ensure the image is 8-bit
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        # Check if the image is not empty
        if gray is None or gray.size == 0:
            print("Empty frame detected")
            continue

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Extract the left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            # Calculate the EAR for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the EAR for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Check if the EAR is below the blink threshold
            if ear < EAR_THRESHOLD:
                COUNTER += 1

                # If the eyes were closed for a sufficient number of frames, sound the alarm
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        winsound.Beep(2500, 1000)  # Beep at 2500 Hz for 1 second
            else:
                COUNTER = 0
                ALARM_ON = False

            # Draw the eye regions
            for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw circles around the eye landmarks

        # Display the resulting frame
        cv2.imshow("Frame", frame)  # Show the frame in a window
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' is pressed
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
```

## Running the Script

To run the script, use the following command:

```sh
python eye_closure_detection.py
```

Make sure to replace `"shape_predictor_68_face_landmarks.dat"` with the actual path to the `.dat` file if it is not in the same directory as your script.

## Notes

- Ensure your webcam is properly connected and accessible.
- Adjust the `EAR_THRESHOLD` and `EAR_CONSEC_FRAMES` constants if necessary to better suit your needs.
- The alarm sound is generated using the `winsound` library, which is specific to Windows. If you are using a different operating system, you may need to use an alternative method for generating the alarm sound.
```
