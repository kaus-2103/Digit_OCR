import cv2
import numpy as np
import os
from PIL import Image



def remove_lines(image, size_threshold=20):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal lines using a kernel
    kernel = np.ones((1, 5), np.uint8)
    lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of the lines
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove horizontal lines based on the size threshold
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > size_threshold:
            image[y:y+h, x:x+w] = [255, 255, 255]  # Set the region to white

    cv2.imshow('Cleaned Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def segment_and_save_digits(image_path, output_folder):
    # Read the input image
    original_image = cv2.imread(image_path)

    # Remove horizontal lines from each digit
    cleaned_original_image = remove_lines(original_image)

    # Convert the cleaned image to grayscale
    gray_image = cv2.cvtColor(cleaned_original_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the contours
    for idx, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract each digit from the original image
        digit = cleaned_original_image[y:y + h, x:x + w]

        # Save each digit as a separate image
        digit_path = os.path.join(output_folder, f'digit_{idx + 1}.png')
        cv2.imwrite(digit_path, digit)

        # Draw a rectangle around the digit on the original image
        cv2.rectangle(cleaned_original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Segmented Digits', cleaned_original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_region(image_path, target_width, target_height):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Define the crop coordinates
            crop_width = 7
            crop_height = height
            crop_left = width - crop_width
            crop_right = width
            
            # Crop the right 7x21 pixels
            cropped_image = img.crop((crop_left, 0, crop_right, crop_height))
            
            # Save the cropped image
            cropped_image.save('test_images/Segmented image/digit_ex1.png')
            
            # Save the remaining part of the image
            remaining_image = img.crop((0, 0, crop_left, crop_height))
            remaining_image.save('test_images/Segmented image/digit_ex2.png')
            
            print("Cropped and saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = 'test_images/imagetest.png'
output_folder = 'test_images/Segmented image'
segment_and_save_digits(image_path, output_folder)

# Replace 'your_image_path.jpg' with the path to your image file
image_path = 'test_images/Segmented image/digit_1.png'
image_size = get_image_size(image_path)

if image_size:
    print(f"Image Size: {image_size[0]} x {image_size[1]} pixels")

# Replace 'your_image_path.jpg' with the path to your image file
image_path = 'test_images/Segmented image/digit_3.png'
target_width = 7
target_height = 21

extract_region(image_path, target_width, target_height)