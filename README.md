MNIST Handwritten Digit Classifier
==================================

An implementation of multilayer neural network using keras.

### About MNIST dataset:
The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.


### Structure of Neural Network:
A neural network is made up by stacking layers of neurons, and is defined by the weights 
of connections and biases of neurons. Activations are a result dependent on a certain input.

* Neural networks are made up of building blocks known as **Sigmoid Neurons**. These are 
named so because their output follows [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function).
* **x<sub>j</sub>** are inputs, which are weighted by **w<sub>j</sub>** weights and the 
neuron has its intrinsic bias **b**. The output of neuron is known as "activation ( **a** )".

![Small Labelled Neural Network](http://i.imgur.com/HdfentB.png)

## Execution:

* Run the `cnn.py` file.
* You can also run the `load_model.py` to skip the computation of NN. It will load the pre saved model from `model.json` and `model.h5` file.

## Result:
![Result of CNN model](https://github.com/aakashjhawar/Handwritten-Digit-Recognition/blob/master/result.png)