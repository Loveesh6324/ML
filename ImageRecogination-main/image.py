from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model('dog_cat_model.h5')  

def preprocess_image(image):
    img = cv2.resize(image,(256, 256))
    img = img.reshape(1, 256, 256, 3)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = preprocess_image(image)
    prediction = model.predict(image)
    if prediction == 0:
        result = 'Cat'
    else:
        result = 'Dog'
        
    
    return render_template('prediction.html', result=result, file=file)

if __name__ == '__main__':
    app.run(debug=True)
