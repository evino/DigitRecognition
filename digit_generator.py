import numpy as np
import matplotlib.pyplot as plt

def GenerateImage(digit, size=(28, 28)):
    """
    Simulate a simple image of a digit.

    Args:
    - digit (int): The digit to simulate in the image (0-9).
    - size (tuple): The size of the image (default 28x28).

    Returns:
    - np.array: The image as a NumPy array.
    """
    # Create an empty image (28x28) with all zeros (black)
    image = np.zeros(size)
    
    # Draw the digit in the middle of the image
    plt.imshow(image, cmap="gray")
    plt.text(12, 12, str(digit), fontsize=12, ha='center', va='center', color='white')
    
    # Save the image to a file
    plt.axis('off')
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the image back as a NumPy array
    img = plt.imread("temp.png")[:, :, 0]
    return img

if __name__ == "__main__":
    # Example usage: generate an image of the digit 5
    img = GenerateImage(7)
    # No need for plt.show() in non-interactive environments
