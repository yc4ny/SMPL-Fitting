import os
import cv2
import argparse
from tqdm import tqdm

def main(input_folder, output_file):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print("No image files found in the input folder.")
        return

    # Sort the image files alphabetically
    image_files.sort()

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as required
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    # Iterate through all image files and write them to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        out.write(img)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an mp4 video from images in a folder.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images')
    parser.add_argument('--output_file', type=str, help='Output mp4 file name')
    args = parser.parse_args()

    main(args.input_folder, args.output_file)
