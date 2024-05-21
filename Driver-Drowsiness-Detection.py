# Library Imports
import cv2
import numpy as np
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance

# Initialize dlib's pre-trained facial landmark detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye landmark indices for left and right eyes based on the 68-point facial landmark model
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    """
    Calculates the eye aspect ratio (EAR) to measure eye openness.
    Args:
        eye (numpy.ndarray): An array of facial landmark coordinates for one eye.
    Returns:
        float: The calculated EAR value.
    """
    # Calculate the distances between key eye landmarks
    X = distance.euclidean(eye[1], eye[5])
    Y = distance.euclidean(eye[2], eye[4])
    Z = distance.euclidean(eye[0], eye[3])
    # Calculate the eye aspect ratio (EAR)
    EAR = (X + Y) / (2.0 * Z)
    return EAR

# Hyperparameters
EAR_THRESHOLD = 0.25 # Threshold limit for Eye Aspect Ratio
flag = 0
DURATION_THRESHOLD = 20 # No of continous flags to be considered as drowsy

# Getting input video source
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Resizing the frame
    frame = imutils.resize(frame, width=450)
    # Convert the grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect multiple faces (if any)
    faces = detect(gray, 0)
    # Iterate over each detected face
    for face in faces:
        # Predict facial landmarks using the pre-trained shape predictor
        shape = predict(gray, face)
        # Convert the predicted shape to a NumPy array for easier manipulation
        shape = face_utils.shape_to_np(shape)
        # Extract the landmark coordinates for the left and right eyes
        lefteye = shape[left_start:left_end]
        righteye = shape[right_start:right_end]
        # Calculate the eye aspect ratio (EAR) for each eye
        leftEAR = eye_aspect_ratio(lefteye)
        rightEAR = eye_aspect_ratio(righteye)
        # Calculate the average EAR for both eyes
        avgEAR = (leftEAR + rightEAR) / 2.0

        # Draw convex hulls around the detected eyes for visualization
        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the average EAR is below the drowsiness threshold
        if avgEAR < EAR_THRESHOLD:
            flag += 1
            # Print a message indicating drowsiness detection and its duration
            print("Drowsiness detected for", flag, "continuous frames")
            # Display an alert message if drowsiness is detected for a prolonged period
            if flag > DURATION_THRESHOLD:
                print("----- Drowsiness Detected - ALERT ! -----")
                cv2.putText(frame, "*** ALERT! ***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "*** ALERT! ***", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,
                else:
            flag = 0
    cv2.imshow("Live Cam - Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()
cap.release()