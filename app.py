import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.keras")  # Change to "model.keras" if needed

# Function to preprocess the input image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)

    # If using binary classification (sigmoid activation)
    if prediction.shape[-1] == 1:
        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    else:  # If using softmax for categorical classification
        label = "Dog" if np.argmax(prediction) == 1 else "Cat"
        confidence = np.max(prediction)
    
    return f"Prediction: {label} ({confidence*100:.2f}%)"

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text"
)

iface.launch(debug=True)  # Launch the interface
