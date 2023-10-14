MNIST Handwritten Digit Classifier
==================================

An implementation of multilayer neural network using keras with an accuracy of 98.314% and using tensorflow with an accuracy over 99%.

### About MNIST dataset:
The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.


### Structure of Neural Network:
A neural network is made up by stacking layers of neurons, and is defined by the weights 
of connections and biases of neurons. Activations are a result dependent on a certain input.

This structure is known as a feedforward architecture because the connections in the network flow forward from the input layer to the output layer without any feedback loops. In this figure:

* The input layer contains the predictors.
* The hidden layer contains unobservable nodes, or units. The value of each hidden unit is some function of the predictors; the exact form of the function depends in part upon the network type and in part upon user-controllable specifications.
* The output layer contains the responses. Since the history of default is a categorical variable with two categories, it is recoded as two indicator variables. Each output unit is some function of the hidden units. Again, the exact form of the function depends in part on the network type and in part on user-controllable specifications.
![Small Labelled Neural Network](http://i.imgur.com/HdfentB.png)


#### Summary of Sequential model


![Summary](https://github.com/aakashjhawar/handwritten-digit-recognition/blob/master/assets/model/model_summary.png)

## Getting Started

How to use
```    
git clone https://github.com/aakashjhawar/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
pip3 install -r requirements.txt 
python3 tf_cnn.py
```
* You can also run the `load_model.py` to skip the training of NN. It will load the pre saved model from `model.json` and `model.h5` files.
```
python3 load_model.py <path/to/image_file>
```
For example
```
python3 load_model.py assets/images/1a.jpg 
```
 
## Prerequisites

- Python 3.5
- OpenCV
```
sudo apt-get install python-opencv
``` 
## Result:
Following image is the prediction of the model.
![Result of CNN model](https://github.com/aakashjhawar/Handwritten-Digit-Recognition/blob/master/result.png)

