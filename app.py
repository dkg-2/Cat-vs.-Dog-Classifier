import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("model.keras") 


def preprocess_image(image):
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0   
    image = np.expand_dims(image, axis=0)  
    return image


def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)

    if prediction.shape[-1] == 1: 
        confidence = prediction[0][0]
        label = "Dog" if confidence > 0.5 else "Cat"
        confidence = confidence if confidence > 0.5 else 1 - confidence
    else:
        confidence = np.max(prediction)
        label = "Dog" if np.argmax(prediction) == 1 else "Cat"

    return label, f"Confidence: {confidence*100:.2f}%"


with gr.Blocks() as demo:
    gr.Markdown("# 🐶🐱 Cat vs. Dog Classifier")
    gr.Markdown("Upload an image, and our AI model will predict whether it's a cat or a dog! 🖼️")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_output = gr.Image(label="Uploaded Image", interactive=False)

    with gr.Row():
        prediction_text = gr.Textbox(label="Prediction", interactive=False)
        confidence_text = gr.Textbox(label="Confidence", interactive=False)

    submit_btn = gr.Button("Predict 🧠")

    def wrapper(image):
        label, confidence = predict(image)
        return image, label, confidence

    submit_btn.click(wrapper, inputs=image_input, outputs=[image_output, prediction_text, confidence_text])


demo.launch()
