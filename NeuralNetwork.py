import numpy as np

class NeuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        # Initialize weights and biases
        self.weights1 = np.random.randn(inputSize, hiddenSize) * 0.01 # Weights for the input -> hidden layer
        self.bias1 = np.zeros((1, hiddenSize))  # Biases for the hidden layer

        self.weights2 = np.random.randn(hiddenSize, outputSize) * 0.01  # Weights for the hidden -> output layer
        self.bias2 = np.zeros((1, outputSize))  # Biases for the output layer

    # Rectified linear unit activation function
    def relu(self, x):
        return np.maximum(0, x)    

    # Softmax activation function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Forward passing through layers
    def forward(self, X):
        # Forward pass: input -> hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # Forward pass: hidden -> output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)  # Output layer (softmax for classification)

        return self.a2
    
    def backpropagation(self, X, y_true, learning_rate=0.01):
        # Compute gradients for backpropagation
        m = X.shape[0]  # Number of samples

        # Gradient of the loss with respect to the output layer
        dA2 = self.a2 - y_true
        dZ2 = dA2  # Derivative of softmax is already incorporated
        dW2 = np.dot(self.a1.T, dZ2) / m
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Gradient for hidden layer
        dA1 = np.dot(dZ2, self.weights2.T)
        dZ1 = dA1 * (self.a1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dZ1) / m
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * dB1
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * dB2

def cross_entropy_loss(y_true, y_pred):
    # Clip predictions to avoid log(0) error and then compute the loss
    epsilon = 1e-12  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

