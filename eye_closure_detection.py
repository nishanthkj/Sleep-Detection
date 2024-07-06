import cv2  # pylint: disable=no-member
import dlib  # pylint: disable=no-member
import numpy as np
from scipy.spatial import distance as dist
import winsound  # For alarm sound on Windows

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Initialize the frame counters and the alarm status
COUNTER = 0
ALARM_ON = False

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()  # pylint: disable=no-member
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # pylint: disable=no-member
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

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

        # Ensure the image is 8-bit
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        winsound.Beep(2500, 1000)  # Beep at 2500 Hz for 1 second
            else:
                COUNTER = 0
                ALARM_ON = False

            # Draw the eye regions
            for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # pylint: disable=no-member

        cv2.imshow("Frame", frame)  # pylint: disable=no-member
        if cv2.waitKey(1) & 0xFF == ord('q'):  # pylint: disable-no-member
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member
