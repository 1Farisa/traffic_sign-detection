# Traffic Sign Classification Project

This project focuses on detecting and classifying traffic signs using a Convolutional Neural Network (CNN) built with Keras and TensorFlow. It includes two main components: a Jupyter notebook for training the model and a Streamlit web application for real-time predictions.

## Files:

### 1. `Traffic_Sign_Classification.ipynb`:
This Jupyter notebook contains the code for training the CNN model. The steps include:
- Loading and preprocessing the dataset (grayscale conversion, equalization, resizing).
- Defining the model architecture using Conv2D and MaxPooling2D layers.
- Applying data augmentation techniques for better generalization.
- Training the model on the traffic sign dataset.
- Saving the trained model as `model.h5`.

### 2. `traffic_sign_streamlit_app.py`:
This Streamlit application allows users to upload an image of a traffic sign and classify it using the trained model. The app includes:
- Image preprocessing (grayscale conversion, equalization, denoising, contrast enhancement, sharpening).
- Loading the trained model and making predictions.
- Displaying the predicted traffic sign class.

## Screenshot of Streamlit App

Here’s a screenshot of the traffic sign classification app in action:

![Streamlit App Screenshot](images/streamlit_app.png)

## Project Overview:

- **Dataset**: The dataset used for training contains 43 classes of traffic signs, with images resized to 32x32 pixels to fit the model input size.
- **Model**: The model is a CNN consisting of Conv2D layers for feature extraction, MaxPooling2D layers for downsampling, and Dense layers for classification. The model is trained using categorical cross-entropy loss and the Adam optimizer.
- **Web Application**: The Streamlit app enables users to upload an image of a traffic sign, which is then classified by the trained model.

## Features:

- **Data Augmentation**: To improve model performance and generalization, data augmentation techniques such as width shift, height shift, zoom, and rotation are applied during training.
- **Real-Time Prediction**: Users can upload an image of a traffic sign, and the app will predict the class in real-time.
- **User-Friendly Interface**: The app provides a simple interface for uploading images and displaying predictions.

## Requirements:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Streamlit
- Matplotlib
- NumPy
- Pandas

## How to Run:

### 1. **Training the Model**:
   - Run the `Traffic_Sign_Classification.ipynb` notebook to train the model and save it as `model.h5`.

### 2. **Running the Streamlit App**:
   - Install Streamlit:  
     ```bash
     pip install streamlit
     ```
   - Run the app:  
     ```bash
     streamlit run traffic_sign_streamlit_app.py
     ```
   - Upload an image of a traffic sign, and the app will predict the class of the sign.

## Sample Output:

After uploading an image, the model will output the predicted class, such as:
- "Speed Limit 50 km/h"
- "Stop"
- "Yield"

## Conclusion:

This project demonstrates the use of deep learning for traffic sign classification, providing a practical tool for real-time prediction with a trained model deployed in a web application.
