import cv2
import numpy as np

def scale_and_crop_center(image, horizontal_scale, vertical_scale, crop_width, crop_height):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * horizontal_scale)
    new_height = int(original_height * vertical_scale)
    
    # Scale the image
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Calculate the crop region
    start_x = (new_width - crop_width) // 2
    start_y = (new_height - crop_height) // 2
    
    # Ensure crop dimensions are within bounds
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    
    # Crop the center part
    cropped_image = scaled_image[start_y:end_y, start_x:end_x]
    
    return cropped_image



def shrink_and_fill(image, shrink_width, shrink_height):
    # Get the original dimensions
    original_height, original_width = image.shape
    
    # Resize the image to the specified smaller dimensions
    shrunk_image = cv2.resize(image, (shrink_width, shrink_height))
    
    # Create a black canvas of the original size
    black_canvas = np.zeros((original_height, original_width), dtype=np.uint8)
    
    # Calculate the top-left corner to center the shrunk image
    start_x = (original_width - shrink_width) // 2
    start_y = (original_height - shrink_height) // 2
    
    # Place the shrunk image on the black canvas
    black_canvas[start_y:start_y + shrink_height, start_x:start_x + shrink_width] = shrunk_image
    
    return black_canvas


def scale_augmentation(image, factor, win_dim):
    return (
        # scale_and_crop_center(image, factor, 1, win_dim[0], win_dim[1]),
        # scale_and_crop_center(image, 1, factor, win_dim[0], win_dim[1]),
        shrink_and_fill(image, image.shape[1], int(image.shape[0]/factor)),
        shrink_and_fill(image, int(image.shape[1]/factor), image.shape[0])
    )