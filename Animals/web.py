from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model= load_model('weight.h5')

# from flask_ngrok import run_with_ngrok

import os

app = Flask(__name__)
# run_with_ngrok(app)
@app.route('/')
def index():
    return render_template('anim.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    print(file)

    filename = secure_filename(file.filename)
    print(filename)
    file.save(os.path.join('static', filename))


    path="static/"+filename
    print(path)
    test_image = image.load_img(path, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(x= test_image)
    print(result)
    if np.argmax(result)  == 0:
      prediction = 'butterfly'
    elif np.argmax(result)  == 1:
      prediction = 'koala'
    elif np.argmax(result)  == 2:
      prediction = 'lion'
    elif np.argmax(result)  == 3:
      prediction = 'panda'
    elif np.argmax(result)  == 4:
      prediction = 'seahorse'
    elif np.argmax(result)  == 5:
      prediction = 'snake'
    elif np.argmax(result)  == 6:
      prediction = 'starfish'
    elif np.argmax(result)  == 7:
      prediction = 'wolf'
    elif np.argmax(result)  == 8:
      prediction = 'woodpecker'
    else:
      prediction = 'zebra'
    
    print(prediction)

    return render_template('anim.html', data=prediction)

if __name__ == '__main__':
    app.run()

    
