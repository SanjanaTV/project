import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model= load_model('weight.h5')
# testing the model
path=r"C:\Users\user\Desktop\cnn_dataset\Horned_Puffin\13.jpg"
#def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (224, 224))
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

