import numpy as np
import cv2



def increase_dilation_percentage(mask, desired_increase_percentage):
    # Convert the boolean mask to uint8 format (0s and 255s)
    mask_uint8 = mask.astype(np.uint8)

    # Initialize kernel size and percentage increase
    kernel_size = 1
    current_increase_percentage = 0

    # Compute the initial dilation to get the baseline area
    dilation = cv2.dilate(mask_uint8, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    original_area = np.count_nonzero(mask)

    # Keep increasing kernel size until desired increase percentage is reached
    while current_increase_percentage < desired_increase_percentage:
        kernel_size += 1
        dilation = cv2.dilate(mask_uint8, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        dilated_area = np.count_nonzero(dilation)
        current_increase_percentage = ((dilated_area - original_area) / original_area) * 100

    return dilation