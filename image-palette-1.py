from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def extract_color_palette(image_path, num_colors=8):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a numpy array for processing
    img_np = np.array(image)

    # Flatten the image array to (num_pixels, num_channels) shape
    pixels = img_np.reshape(-1, 3)

    # Perform K-means clustering to extract dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    return colors


def generate_palette_image(colors):
    # Create a new image to display the palette
    palette_image = Image.new("RGB", (100 * len(colors), 100))

    # Fill the new image with color squares
    for i, color in enumerate(colors):
        color_block = Image.new("RGB", (100, 100), tuple(color))
        palette_image.paste(color_block, (i * 100, 0))

    return palette_image


if __name__ == "__main__":
    input_image_path = "input_image.png"
    output_palette_image_path = "color_palette.png"

    # Step 1: Extract the color palette from the input image
    palette = extract_color_palette(input_image_path, num_colors=8)

    # Step 2: Generate an image displaying the palette
    palette_image = generate_palette_image(palette)
    palette_image.save(output_palette_image_path)
