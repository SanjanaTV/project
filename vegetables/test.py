import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model= load_model('weight.h5')

# testing the model
# def testing_image(image_directory):
path=r"C:\Users\user\Desktop\CNN_Project\Vegitable\Veg\Pumpkin\0056.jpg"
test_image = image.load_img(path, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(x= test_image)
print(result)
if np.argmax(result)  == 0:
  prediction = 'Capsicum'
elif np.argmax(result)  == 1:
  prediction = 'Cauliflower'
elif np.argmax(result)  == 2:
  prediction = 'Pumpkin'
elif np.argmax(result)  == 3:
  prediction = 'Radish'

else:
  prediction = 'Tomato'

print(prediction)

  

