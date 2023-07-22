from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def extract_color_palette(image_path, num_colors=5):
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

    # Create a new image with the extracted palette
    palette_image = Image.new("RGB", (100 * num_colors, 100))
    for i, color in enumerate(colors):
        color_block = Image.new("RGB", (100, 100), tuple(color))
        palette_image.paste(color_block, (i * 100, 0))

    return colors, palette_image


def apply_color_palette(image_path, colors):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a numpy array for processing
    img_np = np.array(image)

    # Flatten the image array to (num_pixels, num_channels) shape
    pixels = img_np.reshape(-1, 3)

    # Perform K-means clustering to map each pixel to the nearest color in the palette
    kmeans = KMeans(n_clusters=len(colors), init=np.array(colors), n_init=1)
    kmeans.fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(int)
    new_img_np = new_colors[kmeans.labels_].reshape(img_np.shape)

    # Create a new image with the applied palette
    new_image = Image.fromarray(np.uint8(new_img_np))

    return new_image


if __name__ == "__main__":
    input_image_path = "input_image.png"
    output_image_path = "output_image.png"
    num_colors = 5

    # Step 1: Extract the color palette from the input image
    palette, palette_image = extract_color_palette(
        input_image_path, num_colors=num_colors)
    palette_image.save("color_palette.png")

    # Step 2: Apply the color palette to the input image and save the output image
    output_image = apply_color_palette(input_image_path, palette)
    output_image.save(output_image_path)
