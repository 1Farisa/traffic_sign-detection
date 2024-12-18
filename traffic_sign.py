import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Define paths for the dataset and labels
path = "C:/Users/USER/Downloads/Dataset"
labelFile = 'C:/Users/USER/Downloads/labels.csv'

# Load the model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Define the class labels
def getClassName(classNo):
    labels = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons', 
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return labels[classNo]

# Image preprocessing with enhanced steps for better clarity
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening kernel
    return cv2.filter2D(img, -1, kernel)

def denoise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)  # Denoising to remove noise

def contrast_enhancement(img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocessing(img):
    # Resize to match the model's expected input size (32x32 for your case)
    img = cv2.resize(img, (32, 32))  # Resize image to 32x32
    img = grayscale(img)
    img = equalize(img)  # Enhance image contrast
    img = denoise(img)  # Remove noise
    img = contrast_enhancement(img)  # Further enhance contrast
    img = sharpen(img)  # Sharpen the image to make details clearer
    img = img / 255.0  # Normalize to range 0-1
    return img

# Predict the class of the uploaded image
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32))  # Resize image to 32x32
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  # Adjust the shape to match the model input
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    preds = getClassName(classIndex[0])
    return preds

# Streamlit app UI
st.title("Traffic Sign Classification")
st.write("Upload an image of a traffic sign, and the model will predict the class.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save the uploaded file to a temporary location
    file_path = os.path.join(path, 'temp_image.jpg')
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make prediction
    result = model_predict(file_path, model)
    
    # Display the result
    st.write(f"Predicted Traffic Sign: {result}")
