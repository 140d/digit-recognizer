# Digit Recognizer System

This project implements a simple neural network for digit recognition using the MNIST dataset. It builds and trains a feedforward neural network using NumPy to recognize handwritten digits from the MNIST dataset. The project uses basic techniques such as forward propagation, backward propagation, ReLU, softmax, and gradient descent to train the model.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)

## Overview

The project builds a simple feedforward neural network with two layers:
1. A hidden layer with ReLU activation.
2. An output layer with softmax activation for multi-class classification.

The model is trained on the MNIST dataset to classify images of handwritten digits (0-9). The dataset is divided into a training set and a validation set (dev set) for model evaluation.

## Requirements

To run this project, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`

These libraries can be installed using `pip`.

## Installation

You can install the required dependencies by running the following command:

```bash
pip install numpy pandas matplotlib
```

## Usage

### 1. Load the Data

The dataset is loaded from a CSV file:

```python
data = pd.read_csv('/kaggle/input/automon/train.csv')
```

### 2. Preprocess the Data

The data is shuffled, normalized, and split into training and validation sets. The image data is normalized by dividing by 255.

### 3. Initialize Parameters

The weight and bias parameters are initialized randomly:

```python
W1 = np.random.rand(10, 784) - 0.5
b1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(10, 1) - 0.5
```

### 4. Forward Propagation

The network uses ReLU activation for the hidden layer and softmax for the output layer:

```python
Z1 = W1.dot(X) + b1
A1 = ReLU(Z1)
Z2 = W2.dot(A1) + b2
A2 = softmax(Z2)
```

### 5. Backward Propagation

The gradients are computed using the chain rule and used to update the weights and biases:

```python
dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
```

### 6. Gradient Descent

The model is trained using gradient descent to minimize the loss:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
```

### 7. Predictions and Evaluation

The model makes predictions using the trained parameters, and the accuracy is evaluated on the dev set:

```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```

### 8. Visualizing Predictions

The trained model is tested on individual images:

```python
test_prediction(0, W1, b1, W2, b2)
```

This will display the image along with the predicted label and the true label.

## Model Details

- **Input Layer**: 784 units (28x28 pixel images flattened into a 1D vector)
- **Hidden Layer**: 10 units with ReLU activation
- **Output Layer**: 10 units with softmax activation (corresponding to digits 0-9)

### Training Process

- The model is trained for 500 iterations using a learning rate of 0.10.
- The model weights are updated using gradient descent.

### Functions Overview

- `init_params()`: Initializes the weights and biases.
- `ReLU()`: Applies the ReLU activation function.
- `softmax()`: Applies the softmax function to the output.
- `forward_prop()`: Performs forward propagation through the network.
- `backward_prop()`: Computes gradients for backpropagation.
- `update_params()`: Updates the weights and biases using gradient descent.
- `get_predictions()`: Returns the predicted labels.
- `get_accuracy()`: Calculates the accuracy of the predictions.
- `gradient_descent()`: Trains the model using gradient descent.
- `make_predictions()`: Makes predictions for the test set.
- `test_prediction()`: Visualizes a test image along with its predicted and true labels.

## Results

The model achieves reasonable accuracy on the dev set. After training, the model can be used to classify handwritten digits from the MNIST dataset. The accuracy on the dev set is printed after training.

## Contributing

If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Any improvements, fixes, or suggestions are welcome!
