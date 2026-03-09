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
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        keypoints = []
                        for landmark in hand_landmarks.landmark:
                            # Append normalized keypoints (x, y, z coordinates)
                            keypoints.append([landmark.x, landmark.y, landmark.z])
                        handedness_label = handedness.classification[0].label  # 'Left' or 'Right'
                        keypoints_video.append((handedness_label, keypoints))  # Store handedness along with keypoints
                else:
                    # Append empty array if no hand detected
                    keypoints_video.append(('Left', np.zeros((21, 3)).tolist()))  # Left hand default
                    keypoints_video.append(('Right', np.zeros((21, 3)).tolist()))  # Right hand default

            keypoints_category[subcategory] = keypoints_video
        
        keypoints_data[category] = keypoints_category

    return keypoints_data

def normalize_keypoints(keypoints):
    normalized_keypoints = []
    for handedness, points in keypoints:
        points_array = np.array(points)
        # Normalize each axis (x, y, z)
        for i in range(3):  # for x, y, z
            min_val = points_array[:, i].min()
            max_val = points_array[:, i].max()
            # Avoid division by zero
            if max_val - min_val > 0:
                points_array[:, i] = (points_array[:, i] - min_val) / (max_val - min_val)
            else:
                points_array[:, i] = 0.0  # If all points are the same, set to 0.0
        normalized_keypoints.append((handedness, points_array.tolist()))  # Keep handedness with normalized points
    return normalized_keypoints

def normalize_json_keypoints(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    normalized_data = {}
    for label, keypoints_list in data.items():
        normalized_data[label] = {}
        for key, keypoints in keypoints_list.items():
            normalized_data[label][key] = normalize_keypoints(keypoints)

    return normalized_data

def main():
    frames_path = r"D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames"  # Folder containing categories like 'Adjectives', 'Pronouns', 'Places'
    keypoints = extract_keypoints_from_frames(frames_path)

    # Save extracted keypoints to JSON
    with open('keypoints1.json', 'w') as f:
        json.dump(keypoints, f, indent=4)

    # Normalize the keypoints from the JSON file
    normalized_data = normalize_json_keypoints('keypoints1.json')

    # Save the normalized data to a new JSON file
    with open('normalized_keypoints1.json', 'w') as f:
        json.dump(normalized_data, f, indent=4)

    print("Normalization complete. Normalized keypoints saved to 'normalized_keypoints1.json'.")

if __name__== "__main__":
    main()

# import cv2
# import os
# import json
# import mediapipe as mp
# import numpy as np

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# def extract_keypoints_from_frames(frames_path):
#     keypoints_data = {}

#     # Iterate over each main category (e.g., 'Adjectives', 'Pronouns', 'Places')
#     for category in os.listdir(frames_path):
#         category_path = os.path.join(frames_path, category)
#         keypoints_category = {}
        
#         # Iterate over each subcategory (e.g., 'clean', 'dirty' in 'Adjectives')
#         for subcategory in os.listdir(category_path):
#             subcategory_path = os.path.join(category_path, subcategory)
#             keypoints_video = []

#             # Iterate over each frame image in the subcategory folder
#             for frame in sorted(os.listdir(subcategory_path)):
#                 frame_path = os.path.join(subcategory_path, frame)
#                 img = cv2.imread(frame_path)

#                 if img is None:
#                     continue

#                 frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 results = hands.process(frame_rgb)

#                 if results.multi_hand_landmarks:
#                     for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                         keypoints = []
#                         for landmark in hand_landmarks.landmark:
#                             # Append normalized keypoints (x, y, z coordinates)
#                             keypoints.append([landmark.x, landmark.y, landmark.z])
#                         handedness_label = handedness.classification[0].label  # 'Left' or 'Right'
#                         keypoints_video.append((handedness_label, keypoints))  # Store handedness along with keypoints
#                 else:
#                     keypoints_video.append(('None', np.zeros((21, 3)).tolist()))  # Append empty array if no hand detected

#             keypoints_category[subcategory] = keypoints_video
        
#         keypoints_data[category] = keypoints_category

#     return keypoints_data

# def normalize_keypoints(keypoints):
#     normalized_keypoints = []
#     for points, handedness in keypoints:
#         points_array = np.array(points)
#         # Normalize each axis (x, y, z)
#         for i in range(3):  # for x, y, z
#             min_val = points_array[:, i].min()
#             max_val = points_array[:, i].max()
#             # Avoid division by zero
#             if max_val - min_val > 0:
#                 points_array[:, i] = (points_array[:, i] - min_val) / (max_val - min_val)
#             else:
#                 points_array[:, i] = 0.0  # If all points are the same, set to 0.0
#         normalized_keypoints.append((handedness, points_array.tolist()))  # Keep handedness with normalized points
#     return normalized_keypoints

# def normalize_json_keypoints(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     normalized_data = {}
#     for label, keypoints_list in data.items():
#         normalized_data[label] = {}
#         for key, keypoints in keypoints_list.items():
#             normalized_data[label][key] = normalize_keypoints(keypoints)

#     return normalized_data


# def main():# Usage
#     frames_path = r"D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames"  # Folder containing categories like 'Adjectives', 'Pronouns', 'Places'
#     keypoints = extract_keypoints_from_frames(frames_path)

#     # Save extracted keypoints to JSON
#     with open('keypoints1.json', 'w') as f:
#         json.dump(keypoints, f, indent=4)

#     # Normalize the keypoints from the JSON file
#     normalized_data = normalize_json_keypoints('keypoints.json')

#     # Save the normalized data to a new JSON file
#     with open('normalized_keypoints1.json', 'w') as f:
#         json.dump(normalized_data, f, indent=4)

#     print("Normalization complete. Normalized keypoints saved to 'normalized_keypoints.json'.")


# if __name__== "__main__":
#     main()