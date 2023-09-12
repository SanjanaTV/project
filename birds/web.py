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
    return render_template('bird_index.html')

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
      prediction = 'Cape_Glossy_Starling'
    elif np.argmax(result)  == 1:
      prediction = 'Cliff_Swallow'
    elif np.argmax(result)  == 2:
      prediction = 'Common_Yellowthroat'
    elif np.argmax(result)  == 3:
      prediction = 'Green_Jay'
    elif np.argmax(result)  == 4:
      prediction = 'Horned_Puffin'
    elif np.argmax(result)  == 5:
      prediction = 'Indigo_Bunting'
    elif np.argmax(result)  == 6:
      prediction = 'Laysan_Albatross'
    elif np.argmax(result)  == 7:
      prediction = 'Red_legged_Kittiwake'
    elif np.argmax(result)  == 8:
      prediction = 'Scarlet_Tanager'

    else:
      prediction = 'White_Pelican'
    
    print(prediction)

    return render_template('bird_index.html', data=prediction)

if __name__ == '__main__':
    app.run()

    
