import cv2
import os

# Set the path to the directory containing your images
images_directory = '/home/eiyike/YOLONERF/results3/video_images/green_rect'

# Set the output video file name
output_video_file = 'output_video.mp4'

# Get the list of image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.jpg') or f.endswith('.png')]

# Sort the files to ensure the correct order
image_files.sort()

# Get the dimensions of the first image to create the video with the same size
first_image = cv2.imread(os.path.join(images_directory, image_files[0]))
height, width, layers = first_image.shape

# Create a VideoWriter object
video = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# Iterate through each image and write it to the video
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    frame = cv2.imread(image_path)
    video.write(frame)

# Release the VideoWriter object
video.release()
