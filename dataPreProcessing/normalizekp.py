import json
import numpy as np

def normalize_keypoints(keypoints):
    normalized_keypoints = []
    for points in keypoints:
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
        normalized_keypoints.append(points_array.tolist())
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

# Usage
input_file = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\keypoints.json'  # Change this to your actual file path
normalized_data = normalize_json_keypoints(input_file)

# Save the normalized data to a new JSON file
with open('normalized_keypoints.json', 'w') as f:
    json.dump(normalized_data, f, indent=4)

print("Normalization complete. Normalized keypoints saved to 'normalized_keypoints.json'.")
