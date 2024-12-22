import cv2
import os

# Define a function to process an image
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, threshInv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # print("thresholding value: {}".format(T))
    dilated = cv2.dilate(threshInv.copy(), None, iterations=2)
    return dilated


if __name__ == "__main__":
    # Input and output directories
    input_dir = "training/1080p"       # Directory containing input images
    output_dir = "processed_images"  # Directory to save processed images
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        # Check if the file is an image (by extension)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {file_name}. Skipping.")
                continue
            # Process the image
            processed_image = process_image(image)
            # Save the processed image to the output directory
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {output_path}")
    print("Batch processing completed.")
