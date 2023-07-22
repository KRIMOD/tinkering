from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def extract_color_palette(image_path, num_colors=8):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (remove alpha channel if exists)
    image = image.convert("RGB")

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


def overlay_palette_on_image(input_image_path, output_image_path, palette_image):
    # Load the input image
    input_image = Image.open(input_image_path)

    # Convert the input image to RGB mode (remove alpha channel if exists)
    input_image = input_image.convert("RGB")

    # Calculate the scale factor for resizing the palette image
    scale_factor = min(input_image.width / palette_image.width,
                       input_image.height / palette_image.height)
    scale_factor = scale_factor * 0.5
    new_width = int(palette_image.width * scale_factor)
    new_height = int(palette_image.height * scale_factor)

    # Resize the palette image
    palette_image_resized = palette_image.resize((new_width, new_height))

    # Calculate the position to center the palette image on the input image
    x_offset = (input_image.width - new_width) // 2
    y_offset = (input_image.height - (2*new_height))

    print(input_image.height)
    print(new_height)
    print(y_offset)

    # Create a new image with the palette overlaid on top of the input image
    output_image = input_image.copy()
    output_image.paste(palette_image_resized, (x_offset, y_offset))

    # Save the final output image
    output_image.save(output_image_path)


if __name__ == "__main__":
    input_image_path = "input_image.png"
    output_palette_image_path = "color_palette.png"
    output_final_image_path = "output_image.png"

    # Step 1: Extract the color palette from the input image
    palette = extract_color_palette(input_image_path, num_colors=8)

    # Step 2: Generate an image displaying the palette
    palette_image = generate_palette_image(palette)
    palette_image.save(output_palette_image_path)

    # Step 3: Overlay the palette on the input image and save the final output image
    overlay_palette_on_image(
        input_image_path, output_final_image_path, palette_image)
