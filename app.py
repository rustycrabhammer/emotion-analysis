import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# Set up the Streamlit page
st.set_page_config(page_title="Face and Emotion Detection", layout="wide")
st.title("Face and Emotion Detection App")

# Initialize face and eye detection
@st.cache_resource
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    if not all([face_cascade.empty() == False, eye_cascade.empty() == False, smile_cascade.empty() == False]):
        st.error("Error: Could not load cascade classifiers")
        return None, None, None
        
    return face_cascade, eye_cascade, smile_cascade

face_cascade, eye_cascade, smile_cascade = load_cascades()

def detect_emotion(face_img, eyes, smile):
    """Simple emotion detection based on eyes and smile"""
    if len(smile) > 0 and len(eyes) >= 2:
        return "Happy"
    elif len(eyes) >= 2:
        return "Neutral"
    else:
        return "Unknown"

def process_image(image):
    # Convert to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Create a copy for drawing
    img_with_detections = img_array.copy()
    
    # Face counter
    face_count = 0
    emotion_stats = {"Happy": 0, "Neutral": 0, "Unknown": 0}
    
    # Process each face
    for (x, y, w, h) in faces:
        face_count += 1
        
        # Get the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_with_detections[y:y+h, x:x+w]
        
        # Detect eyes and smile in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        
        # Detect emotion
        emotion = detect_emotion(roi_gray, eyes, smile)
        emotion_stats[emotion] += 1
        
        # Draw rectangles and labels
        cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_with_detections, f'Face: {emotion}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Draw rectangles for eyes and smile
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    return img_with_detections, face_count, emotion_stats

# Create sidebar for options
st.sidebar.title("Options")
detection_mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam"])

if detection_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process the image and display results
        with col2:
            st.subheader("Processed Image")
            processed_img, face_count, emotion_stats = process_image(image)
            st.image(processed_img, use_column_width=True)
        
        # Display statistics
        st.subheader("Detection Results")
        st.write(f"Faces detected: {face_count}")
        
        if face_count > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Happy", emotion_stats["Happy"])
            col2.metric("Neutral", emotion_stats["Neutral"])
            col3.metric("Unknown", emotion_stats["Unknown"])

elif detection_mode == "Webcam":
    run_webcam = st.sidebar.button("Start Webcam")
    stop_webcam = st.sidebar.button("Stop Webcam")
    
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if run_webcam:
        st.session_state.webcam_running = True
    
    if stop_webcam:
        st.session_state.webcam_running = False
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if st.session_state.webcam_running:
        try:
            # Create a video capture object
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please make sure your webcam is connected and not being used by another application.")
            else:
                while st.session_state.webcam_running:
                    # Read frame from webcam
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture image from webcam.")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # Process the frame
                    processed_img, face_count, emotion_stats = process_image(image)
                    
                    # Display the processed frame
                    frame_placeholder.image(processed_img, caption="Webcam Feed", use_column_width=True)
                    
                    # Display statistics
                    with stats_placeholder.container():
                        st.write(f"Faces detected: {face_count}")
                        if face_count > 0:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Happy", emotion_stats["Happy"])
                            col2.metric("Neutral", emotion_stats["Neutral"])
                            col3.metric("Unknown", emotion_stats["Unknown"])
                    
                    # Add a small sleep to reduce CPU usage
                    time.sleep(0.1)
                    
                    # Check if the webcam should still be running
                    if not st.session_state.webcam_running:
                        break
                
                # Release the webcam
                cap.release()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.webcam_running = False

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app demonstrates face and emotion detection using OpenCV and Streamlit.
    
    Features:
    - Face detection
    - Eye detectdfdion
    - Smile detection
    - Basic dfemotion classification
    """
)