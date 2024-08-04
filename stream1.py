import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import PIL.Image
from io import BytesIO
import tempfile
import os
import threading

# Initialize MediaPipe models
mp_objectron = mp.solutions.objectron
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helper function to convert OpenCV image to PIL image
def opencv_to_pil(image):
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Helper function to resize image
def resize_image(image, width=800, height=600):
    return image.resize((width, height))

# Function to process frames with MediaPipe models
def process_frame(image, model):
    if model == "Objectron":
        objectron = mp_objectron.Objectron(model_name='Cup')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = objectron.process(image_rgb)
        annotated_image = image.copy()
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(annotated_image, detected_object.rotation, detected_object.translation)
    elif model == "Holistic":
        holistic_model = mp_holistic.Holistic()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(image_rgb)
        annotated_image = image.copy()
        if results.face_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    elif model == "Hands":
        hands = mp_hands.Hands()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    elif model == "Face Mesh":
        face_mesh = mp_face_mesh.FaceMesh()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        annotated_image = image.copy()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    elif model == "Pose":
        pose = mp_pose.Pose()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    elif model == "Face Detection":
        face_detection = mp_face_detection.FaceDetection()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        annotated_image = image.copy()
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
    
    return resize_image(opencv_to_pil(annotated_image))

# Function to process video asynchronously
def process_video(video_path, model, frame_rate, output_queue):
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break
        
        # Process frame with selected model
        processed_frame = process_frame(frame, model)
        
        # Add processed frame to queue
        output_queue.append(processed_frame)
        
        # Skip frames to match desired frame rate
        for _ in range(frame_interval - 1):
            video_cap.grab()
    
    video_cap.release()

# Main Streamlit app
def main():
    st.title("MediaPipe Streamlit App")

    # Sidebar for model selection and control
    model = st.sidebar.selectbox("Choose Model", [
        "Objectron",
        "Holistic",
        "Hands",
        "Face Mesh",
        "Pose",
        "Face Detection"
    ])
    
    # Select input source
    input_source = st.sidebar.radio("Select Input Source", ("Webcam", "Upload Video"))
    frame_rate = st.sidebar.slider("Frame Rate", 1, 30, 15)

    if input_source == "Webcam":
        run_button = st.sidebar.button("Run Stream")
        stop_button = st.sidebar.button("Stop Stream")
        
        if run_button:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.error("Failed to capture image")
                    break
                
                # Process frame with selected model
                output_image = process_frame(frame, model)
                
                # Display image in Streamlit
                stframe.image(output_image, channels="RGB", use_column_width=True)
                
                # Check if "Stop Stream" button was pressed
                if stop_button:
                    stframe.empty()
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            # Release the video capture and destroy windows
            cap.release()
            cv2.destroyAllWindows()
    
    elif input_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Save the uploaded video file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Queue to hold processed frames
            output_queue = []
            # Start video processing in a separate thread
            processing_thread = threading.Thread(target=process_video, args=(temp_file_path, model, frame_rate, output_queue))
            processing_thread.start()
            
            stframe = st.empty()
            while processing_thread.is_alive() or output_queue:
                if output_queue:
                    # Display the most recent processed frame
                    stframe.image(output_queue.pop(0), channels="RGB", use_column_width=True)
            
            # Ensure processing thread has completed
            processing_thread.join()
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
