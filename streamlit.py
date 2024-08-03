import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_objectron = mp.solutions.objectron
mp_hands = mp.solutions.hands

# Define functions for different functionalities
def objectron_detection(image):
    with mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, model_name='Cup') as objectron:
        results = objectron.process(image)
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
    return image

def holistic_detection(image):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
        results = holistic_model.process(image)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    return image

def hand_landmarks(image):
    with mp_hands.Hands() as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image

def face_mesh(image):
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    return image

def pose_detection(image):
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

def face_detection(image):
    with mp_face_detection.FaceDetection() as face_detection:
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
    return image

# Streamlit app
st.title("Computer Vision Application")
option = st.selectbox(
    "Select functionality",
    ("Objectron Detection", "Holistic Detection", "Hand Landmarks", "Face Mesh", "Pose Detection", "Face Detection")
)

run = st.checkbox('Run')

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        # Resize the frame to speed up processing
        frame = cv2.resize(frame, (640, 480))

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform selected functionality
        if option == "Objectron Detection":
            frame = objectron_detection(frame)
        elif option == "Holistic Detection":
            frame = holistic_detection(frame)
        elif option == "Hand Landmarks":
            frame = hand_landmarks(frame)
        elif option == "Face Mesh":
            frame = face_mesh(frame)
        elif option == "Pose Detection":
            frame = pose_detection(frame)
        elif option == "Face Detection":
            frame = face_detection(frame)

        # Convert the frame back to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        stframe.image(frame, channels="BGR")

    cap.release()
