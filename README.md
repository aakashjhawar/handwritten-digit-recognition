MNIST Handwritten Digit Classifier
==================================

An implementation of multilayer neural network using keras.

### About MNIST dataset:
The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

### Brief Background:


![Sigmoid Neuron](http://i.imgur.com/dOkT9Y9.png)

* Neural networks are made up of building blocks known as **Sigmoid Neurons**. These are 
named so because their output follows [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function).
* **x<sub>j</sub>** are inputs, which are weighted by **w<sub>j</sub>** weights and the 
neuron has its intrinsic bias **b**. The output of neuron is known as "activation ( **a** )".

_**Note:** There are other functions in use other than sigmoid, but this information for
now is sufficient for beginners._

* A neural network is made up by stacking layers of neurons, and is defined by the weights 
of connections and biases of neurons. Activations are a result dependent on a certain input.


### Structure of Neural Network:

I have followed a particular convention in indexing quantities.
Dimensions of quantities are listed according to this figure.

![Small Labelled Neural Network](http://i.imgur.com/HdfentB.png)


#### **Layers**
* Input layer is the **0<sup>th</sup>** layer, and output layer 
is the **L<sup>th</sup>** layer. Number of layers: **N<sub>L</sub> = L + 1**.
```
sizes = [2, 3, 1]
```

#### **Weights**
* Weights in this neural network implementation are a list of 
matrices (`numpy.ndarrays`). `weights[l]` is a matrix of weights entering the 
**l<sup>th</sup>** layer of the network (Denoted as **w<sup>l</sup>**).  
* An element of this matrix is denoted as **w<sup>l</sup><sub>jk</sub>**. It is a 
part of **j<sup>th</sup>** row, which is a collection of all weights entering 
**j<sup>th</sup>** neuron, from all neurons (0 to k) of **(l-1)<sup>th</sup>** layer.  
* No weights enter the input layer, hence `weights[0]` is redundant, and further it 
follows as `weights[1]` being the collection of weights entering layer 1 and so on.
```
weights = |¯   [[]],    [[a, b],    [[p],   ¯|
          |              [c, d],     [q],    |
          |_             [e, f]],    [r]]   _|
```

#### **Biases**
* Biases in this neural network implementation are a list of one-dimensional 
vectors (`numpy.ndarrays`). `biases[l]` is a vector of biases of neurons in the 
**l<sup>th</sup>** layer of network (Denoted as **b<sup>l</sup>**).  
* An element of this vector is denoted as **b<sup>l</sup><sub>j</sub>**. It is a 
part of **j<sup>th</sup>** row, the bias of **j<sup>th</sup>** in layer.  
* Input layer has no biases, hence `biases[0]` is redundant, and further it 
follows as `biases[1]` being the biases of neurons of layer 1 and so on.
```
biases = |¯   [[],    [[0],    [[0]]   ¯|
         |     []],    [1],             |
         |_            [2]],           _|
```

#### **'Z's**
* For input vector **x** to a layer **l**, **z** is defined as: 
**z<sup>l</sup> = w<sup>l</sup> . x + b<sup>l</sup>**
* Input layer provides **x** vector as input to layer 1, and itself has no input, 
weight or bias, hence `zs[0]` is redundant.
* Dimensions of `zs` will be same as `biases`.

#### **Activations**
* Activations of **l<sup>th</sup>** layer are outputs from neurons of **l<sup>th</sup>** 
which serve as input to **(l+1)<sup>th</sup>** layer. The dimensions of `biases`, `zs` and 
`activations` are similar.
* Input layer provides **x** vector as input to layer 1, hence `activations[0]` can be related 
to **x** - the input training example.

#### **Execution of Neural network**
```
#to train and test the neural network algorithm, please use the following command
python main.py
```