import random
import cv2
import numpy as np


def extract_random_clip(filename, duration):
    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = random.randint(0, int(total_frames - duration * fps))
    end_frame = start_frame + int(duration * fps)

    clips = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if ret:
            clips.append(frame)
        else:
            break

    cap.release()
    return clips


def merge_clips(clips, audio_file, output_file):
    height, width, _ = clips[0].shape

    # Read the audio file
    audio = cv2.VideoCapture(audio_file)
    audio_fps = audio.get(cv2.CAP_PROP_FPS)

    # Create a video writer with audio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, audio_fps, (width, height))

    # Write each frame with audio
    for i, frame in enumerate(clips):
        out.write(frame)

    # Release resources
    audio.release()
    out.release()


clip_count = 5  # Number of clips to extract
clip_duration = 5  # Duration of each clip in seconds
file_name = 'The.shining.1980.720p.x264.mkv'

clips = []
for i in range(clip_count):
    clip = extract_random_clip(file_name, clip_duration)
    clips.extend(clip)

merge_clips(clips, file_name, 'output_movie.mp4')
