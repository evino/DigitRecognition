import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from digit_generator import generate_image
from NeuralNetwork import NeuralNetwork, cross_entropy_loss

# Neural Network Parameters
inputSize = 784 # 28x28 pixels flattened
hiddenSize = 1000 # Hidden layer size
outputSize = 10 # 10 digits (0-9)
epochs = 1000
learningRate = 0.01

# Create Neural Network Object
nn = NeuralNetwork(inputSize, hiddenSize, outputSize)

# Generate training data (X: images, y: labels)
X_train = []
y_train = []

numDigits = 10
numImagesPerDigit = 100
print("DEBUG1")

# Generate some training data (X: images, y: labels)
for digit in range(0, numDigits):
    for i in range(numImagesPerDigit):
        # Generate noisy digit image
        # img = generate_image(digit, noise_factor=0.1).flatten()  # Flatten to 1D array (784 Pixels)
        img = generate_image(digit).flatten()  # Flatten to 1D array (784 Pixels)
        # print(img.size)
        # print(f"Generated image shape: {img.shape}")
        X_train.append(img)

        # One-hot encoded label
        label = np.zeros(numDigits)
        label[digit] = 1
        y_train.append(label)
    print(f"Done with digit {digit}")

# Convert to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize pixel values
X_train = X_train / 255.0

print("DEBUG2")

# Training loop
for epoch in range(epochs):
    # Forward Pass
    y_pred = nn.forward(X_train)

    # Compute the loss
    loss = cross_entropy_loss(y_train, y_pred)

    # Back-propogation and update weights
    nn.backpropagation(X_train, y_train, learningRate)

    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss}")


# Test output
testImg = generate_image(4).flatten()

# Get model's prediction
testPrediction = nn.forward(testImg)
print(f"tesPred is {testPrediction}")
predictedDigit = np.argmax(testPrediction)
print(f"Predicted Digit is {predictedDigit}")
