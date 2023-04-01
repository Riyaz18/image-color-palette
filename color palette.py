import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

def generate_palette():
    # Prompt user for image path
    image_path = input("Enter the path to the image: ")

    # Load image
    image = cv2.imread(image_path)
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten the image
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Prompt user for number of colors
    k = int(input("Enter the number of colors in the palette: "))

    # Run KMeans on the flattened image
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    # Get the colors from the KMeans model
    colors = kmeans.cluster_centers_.astype(int)

    # Plot the colors
    plt.figure(figsize=(8, 2))
    plt.imshow([colors], interpolation='nearest')
    plt.axis('off')
    plt.show()

    return colors

generate_palette()
