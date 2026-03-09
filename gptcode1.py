import cv2
import os
import numpy as np
from imgaug import augmenters as iaa

# Path to the videos directory
base_path = 'D:\FINAL_YEAR_PROJECT\Gesture2Speech\videos'

# Define label mapping based on folder structure
label_mapping = {
    "he": 0, "i": 1, "she": 2, "they": 3, "we": 4, "you": 5,
    "clean": 6, "dirty": 7, "strong": 8, "weak": 9,
    "boy": 10, "girl": 11,
    "hospital": 12, "house": 13, "school": 14, "university": 15,
}

# Augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    
    iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND channel.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

def process_videos(base_path):
    # Loop through each category folder
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        # Loop through each video file
        for video_name in os.listdir(category_path):
            video_path = os.path.join(category_path, video_name)
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Frame operations
                frame_count += 1
                frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224
                images_aug = seq(images=[frame])  # Apply augmentation
                label = label_mapping[category]  # Get label from folder name
                # Save frame
                save_path = os.path.join('D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames', f"{category}_{video_name}_{frame_count}.png")
                cv2.imwrite(save_path, images_aug[0])
            cap.release()

process_videos(base_path)
