# later...
# - - - - - - - LOAD THE MODEL - - - - - - - -
 
# load json and create model
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
import cv2
image_test = cv2.imread('dataset/6c.jpg', 0)      # Predict images from dataset folder
image_test_1 = cv2.resize(image_test, (28,28))    # For plt.imshow
image_test_2 = image_test_1.reshape(1,1,28,28)    # For input to model.predict_classes

#cv2.imshow('number', image_test_1)
loaded_model_pred = loaded_model.predict_classes(image_test_2, verbose = 0)
print('Prediction of loaded_model: {}'.format(loaded_model_pred[0]))