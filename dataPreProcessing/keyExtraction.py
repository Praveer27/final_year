import cv2
import os
import json
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_keypoints_from_frames(frames_path):
    keypoints_data = {}

    # Iterate over each main category (e.g., 'Adjectives', 'Pronouns', 'Places')
    for category in os.listdir(frames_path):
        category_path = os.path.join(frames_path, category)
        keypoints_category = {}
        
        # Iterate over each subcategory (e.g., 'clean', 'dirty' in 'Adjectives')
        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)
            keypoints_video = []

            # Iterate over each frame image in the subcategory folder
            for frame in sorted(os.listdir(subcategory_path)):
                frame_path = os.path.join(subcategory_path, frame)
                img = cv2.imread(frame_path)

                if img is None:
                    continue

                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        keypoints = []
                        for landmark in hand_landmarks.landmark:
                            # Append normalized keypoints (x, y, z coordinates)
                            keypoints.append([landmark.x, landmark.y, landmark.z])
                        keypoints_video.append(keypoints)
                else:
                    keypoints_video.append([np.zeros((21, 3)).tolist()])  # Append empty array if no hand detected

            keypoints_category[subcategory] = keypoints_video
        
        keypoints_data[category] = keypoints_category

    return keypoints_data

# Call the function to extract keypoints from the frames folder
frames_path = r"D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames"  # Folder containing categories like 'Adjectives', 'Pronouns', 'Places'
keypoints = extract_keypoints_from_frames(frames_path)

# Save keypoints to JSON
with open("keypoints.json", "w") as f:
    json.dump(keypoints, f)
