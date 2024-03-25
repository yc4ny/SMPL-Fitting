import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def horizontal_concat_images(img1, img2):
    # Get the dimensions of the images
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    # Determine the maximum height
    max_height = max(height1, height2)

    # Calculate the width to maintain aspect ratio for img2
    new_width2 = int((max_height / height2) * width2)

    # Resize img1 to match max_height and keep its original width
    img1_resized = cv2.resize(img1, (int(width1 * (max_height / height1)), max_height))

    # Resize img2 to match max_height while keeping its aspect ratio
    img2_resized = cv2.resize(img2, (new_width2, max_height))

    # Concatenate images horizontally
    concat_img = cv2.hconcat([img1_resized, img2_resized])

    return concat_img

def main():
    parser = argparse.ArgumentParser(description='Horizontally concatenate images from two directories.')
    parser.add_argument('--dir1', type=str, help='Path to the first directory containing images')
    parser.add_argument('--dir2', type=str, help='Path to the second directory containing images')
    parser.add_argument('--output', type=str, default='output', help='Output directory for concatenated images')
    args = parser.parse_args()

    images1 = load_images_from_folder(args.dir1)
    images2 = load_images_from_folder(args.dir2)

    if len(images1) != len(images2):
        print("Error: Number of images in directories do not match.")
        return

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i, (img1, img2) in tqdm(enumerate(zip(images1, images2))):
        concatenated_image = horizontal_concat_images(img1, img2)
        output_path = os.path.join(args.output, f"{i:05d}.jpg")
        cv2.imwrite(output_path, concatenated_image)

if __name__ == "__main__":
    main()
