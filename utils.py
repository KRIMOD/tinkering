def convert_timestamp(timestamp_seconds):
  hours = int(timestamp_seconds // 3600)
  minutes = int((timestamp_seconds % 3600) // 60)
  seconds = int(timestamp_seconds % 60)

  return f"{hours:02d}_{minutes:02d}_{seconds:02d}"


# def get_dominant_color_from_frame(frame_path):
#   print(f"Calculating the dominant colors for the frame '{frame_path}'")
#    # Read the frame
#   frame = cv2.imread(frame_path)

#   # Reshape the frame to a 2D array of pixels
#   pixels = frame.reshape(-1, 3)

#   # Perform color quantization using K-means clustering
#   num_colors = 3  # Adjust the number of colors based on your needs
#   kmeans = KMeans(n_clusters=num_colors, n_init=10)
#   kmeans.fit(pixels)

#   # Get the dominant colors and their frequencies
#   colors, counts = np.unique(kmeans.labels_, return_counts=True)

#   # Find the index of the most frequent color
#   leading_color_index = np.argmax(counts)

#   # Get the BGR values of the leading color
#   leading_color = kmeans.cluster_centers_[leading_color_index]
#   bgr_values = leading_color.astype(int)

#   # Convert the BGR values to RGB
#   rgb_values = bgr_values[::-1]
  
#   return rgb_values
