import cv2
import sys
from keras.models import model_from_json

def predict_digit(image_path):
    # ----- LOAD SAVED MODEL -----
    json_file = open('model.json', 'r')     
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk.")

    image = cv2.imread(image_path, 0)      
    image1 = cv2.resize(image, (28,28))    # For cv2.imshow: dimenstions=[28][28]
    image2 = image1.reshape(1,1,28,28) 
    cv2.imshow('digit', image1 )
    loaded_model_pred = loaded_model.predict_classes(image2, verbose = 0)
    return loaded_model_pred[0]    

def main(image_path):
    predicted_digit = predict_digit(image_path)
    print('Predicted Digit: {}'.format(predicted_digit))
 
if __name__ == "__main__":
    try:
        main(image_path = sys.argv[1])
    except:
        print('[ERROR]: Image not found')
    #    try:
#        main(image_path = sys.argv[1])
#    except:             #    except IndexError:
#        print('[ERROR]: Image not found')