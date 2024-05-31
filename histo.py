import cv2  # Importing OpenCV
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Function to display histogram of an image
def display_histogram(image_path):
    # Read the image in grayscale mode
    img = cv2.imread(image_path, 0)
    
    # Check if the image is loaded successfully
    if img is None:
        print("Error: Could not open or find the image.")
        return
    
    # Calculate the histogram using OpenCV
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Plot the histogram using matplotlib
    plt.figure()
    plt.plot(histg, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Path to the image file
# image_path = 'image_with_watermark.jpg'
image_path = 'yuldane.jpg'
# image_path = 'wm.jpg'
# image_path = 'recovered_watermark.jpg'


# Display the histogram
display_histogram(image_path)
