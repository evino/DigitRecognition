import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_image(digit, size=(28, 28)):
    """
    Generate a clean simulated image of a digit.

    Args:
    - digit (int): The digit to simulate (0-9).
    - size (tuple): The size of the final image (default 28x28).

    Returns:
    - np.array: The image as a NumPy array of shape (28, 28).
    """
    # Create a blank figure with no axes, set size to match target output (28x28)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis('off')  # Turn off axis

    # Draw the digit in the center
    ax.text(
        1.5, 1.5, str(digit),
        fontsize=40, ha='center', va='center', color='white'
    )

    # Render the figure to a canvas
    plt.draw()

    # Extract the canvas buffer as RGBA
    canvas = fig.canvas
    image_data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

    # Get canvas dimensions and reshape the buffer
    width, height = canvas.get_width_height()
    image = image_data.reshape((height, width, 4))  # ARGB format (4 channels)

    # Convert to grayscale by taking the red channel (or average of RGB)
    image = image[:, :, 1] / 255.0  # Normalize to [0, 1] (take green channel here for grayscale)

    # Resize to the target size (28x28) using PIL
    image = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to 8-bit image
    image = image.resize(size, Image.Resampling.LANCZOS)  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize back to [0, 1]

    plt.close(fig)  # Close the figure to free resources

    return image
