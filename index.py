import sys
import os
import cv2

import numpy as np
from PIL import Image

from utils import convert_timestamp


def generate_frames(video_path, num_of_frames, start_time=1.0):
    video_path = video_path.replace('\\', '\\\\')
    output_dir_name = "_".join(video_path.split('\\')[-1].split(".")[0:-1])

    cam = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (fps) of the video
    fps = cam.get(cv2.CAP_PROP_FPS)

    # Calculate the duration of the video in seconds
    duration_seconds = int(total_frames / fps)

    try:
        # creating a folder named data
        if not os.path.exists(output_dir_name+'/frames_'+str(num_of_frames)):
            os.makedirs(output_dir_name+'/frames_'+str(num_of_frames))

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    # Set the desired timestamp
    timestamp_seconds = start_time  # Replace with your desired timestamp in seconds

    # Step between each frame
    step = int(duration_seconds / num_of_frames)

    if start_time > step:
        raise Exception("Choose a smaller start time")

    while (currentframe < num_of_frames):
        # Set video's current position to the desired timestamp
        cam.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './'+output_dir_name + '/frames_' + \
                str(num_of_frames)+'/frame_' + \
                convert_timestamp(timestamp_seconds) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            timestamp_seconds += step
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def get_dominant_color_from_frame(frame_path, palette_size):

    pil_img = Image.open(frame_path)
    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]

    return dominant_color


def generate_canvas_from_frames(video_path, image_width, image_height, num_of_frames, palette_size=16):
    video_path = video_path.replace('\\', '\\\\')
    output_dir_name = "_".join(video_path.split('\\')[-1].split(".")[0:-1])

    colors = []
    folder_path = output_dir_name+'/frames_'+str(num_of_frames)
    file_names = os.listdir(folder_path)

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            colors.append(get_dominant_color_from_frame(
                file_path, palette_size))

    num_columns = len(colors)

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Calculate the width of each column
    column_width = image_width // num_columns

    # Draw each column with a different color
    for i in range(num_columns):
        start_col = i * column_width
        end_col = (i + 1) * column_width
        image[:, start_col:end_col] = colors[i]

    # Save the image as PNG
    output_path = output_dir_name+'/canvas_n_' + \
        str(num_columns)+'p_' + str(palette_size) + '.png'
    cv2.imwrite(output_path, image)

    print(
        f"Image with {num_columns} columns of different colors generated and saved as '{output_path}'.")


# video_path="samples\\One.Flew.Over.The.Cuckoo's.Nest.1080p.BrRip.x264.YIFY.mp4"
source_video_path = sys.argv[1]
# num_of_frames = int(sys.argv[2])
num_of_frames_arr = [16, 32, 64, 128, 256, 512, 1024]

# Define image width, height, and number of columns
image_width = 2048
image_height = 2048  # Adjust the height as needed

for num_of_frames in num_of_frames_arr:
    generate_frames(source_video_path, num_of_frames, 5.0)
    generate_canvas_from_frames(
        source_video_path, image_width, image_height, num_of_frames, palette_size=16)
