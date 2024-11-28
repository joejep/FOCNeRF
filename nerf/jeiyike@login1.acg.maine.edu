

from moviepy.editor import *
import cv2
import os

# Specify the path to the folder containing the images
folder_path = '/home/eiyike/FASTER_ALL_RAYS/movie_images'

# Initialize an empty list to store the images
images = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image file extensions
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the image
        image = cv2.imread(file_path)
        
        if image is not None:
            # Append the image to the list
            images.append(image)
        else:
            print(f"Failed to load image: {filename}")

clip=ImageSequenceClip(images,fps=1)
# clip.write_gif("circle.gif",fps=15)
# breakpoint()
clip.write_videofile("white_bg.mp4",fps=1)

