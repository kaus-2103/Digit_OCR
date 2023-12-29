import cv2
from train_model import new,view
import os




if __name__ == '__main__':
    folder_path = 'test_images/Segmented image/'
    padd = 10

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Process the image using the new and view functions
            new(image, padd)
            view(image, padd)


