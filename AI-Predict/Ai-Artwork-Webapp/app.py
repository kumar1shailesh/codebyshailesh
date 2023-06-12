from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

app = Flask(__name__)

# URL for the pre-trained Fast Neural Style Transfer model
MODEL_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed'

# Load the pre-trained model
hub_model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=[None, None, 3]),
                                 hub.KerasLayer(MODEL_URL)])

def preprocess_image(image):
    # Preprocess the image for the model
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def stylize_image(content_image):
    # Preprocess the content image
    content_image = preprocess_image(content_image)

    # Stylize the image using the model
    stylized_image = hub_model(tf.constant(content_image))[0]
    
    # Convert the stylized image to PIL format
    stylized_image = (stylized_image * 255).numpy().astype(np.uint8)
    stylized_image = Image.fromarray(stylized_image)

    return stylized_image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        if image_file:
            # Read the image file
            image = Image.open(image_file)

            # Stylize the image
            stylized_image = stylize_image(image)

            # Save the stylized image
            stylized_image_path = 'static/stylized_image.jpg'
            stylized_image.save(stylized_image_path)

            return redirect('/result')

    return render_template('index.html')

@app.route('/result')
def display_result():
    stylized_image_path = 'static/stylized_image.jpg'
    return render_template('result.html', stylized_image=stylized_image_path)

if __name__ == '__main__':
    app.run(debug=True)
