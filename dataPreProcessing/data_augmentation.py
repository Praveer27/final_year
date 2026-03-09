import json
import random
import os

def load_keypoints_json(filepath):
    """ Load the JSON file containing keypoints data. """
    with open(filepath, 'r') as file:
        return json.load(file)

def replace_empty_with_default(keypoints_frame, default_value=[0.0, 0.0, 0.0]):
    """ Replace empty arrays in a keypoints frame with the default value. """
    return [kp if len(kp)!=0 else default_value for kp in keypoints_frame]

def process_keypoints(keypoints_list):
    """ Process keypoints by replacing empty arrays and sampling 70% of the keypoints. """
    output_keypoints_list = []
    for keypoints_frame in keypoints_list:
        # print(keypoints_frame)
        # Ensure no empty arrays exist, replace with [0.0, 0.0, 0.0]
        # keypoints_frame = replace_empty_with_default(keypoints_frame, default_value=[0.0, 0.0, 0.0])

        # Check if the frame is not empty after replacement
        if len(keypoints_frame)!=0:
            num_points_to_select = int(0.7 * len(keypoints_frame))
            random_points = random.sample(keypoints_frame, num_points_to_select)
        else:
            random_points = [[0.0, 0.0, 0.0]]  # If frame is empty, insert a default point
        
        # Ensure random_points also don't have any empty lists
        if len(random_points)==0:
            random_points=[[0.0,0.0,0.0]]
        output_keypoints_list.append(random_points)
            # print("***")

    return output_keypoints_list

def main():
    # Path to the normalized keypoints JSON file
    input_file = r'/Users/ranjannaik/Desktop/FINAL_YEAR_PROJECT/Gesture2Speech/dataPreProcessing/normalized_keypoints.json'
    
    # Load keypoints data from the JSON file
    try:
        keypoints_data = load_keypoints_json(input_file)
        print(f"Loaded keypoints from {input_file}.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # Get the base folder name from the input file (e.g., normalized_keypoints)
    base_folder = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(base_folder, exist_ok=True)
    print(f"Base folder created: {base_folder}")

    # Iterate over each category (like Adjectives, Pronouns, etc.)
    for category, labels_data in keypoints_data.items():
        category_folder = os.path.join(base_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        print(f"Category folder created: {category_folder}")

        # Iterate over each label (like clean, dirty, etc.)
        for label, keypoints_list in labels_data.items():
            label_folder = os.path.join(category_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            print(f"Label folder created: {label_folder}")

            # Prepare the path for the output keypoints.json file
            output_path = os.path.join(label_folder, 'keypoints.json')

            # Process keypoints and randomly select 70% per frame, replacing empty arrays
            output_keypoints_list = process_keypoints(keypoints_list)

            # Write the processed keypoints to keypoints.json
            if output_keypoints_list:
                try:
                    with open(output_path, 'w') as outfile:
                        json.dump({label: output_keypoints_list}, outfile)
                    print(f"File saved: {output_path}")
                except Exception as e:
                    print(f"Error writing to {output_path}: {e}")
            else:
                print(f"Keypoints list is empty for {label}. Skipping writing to {output_path}.")

# Corrected condition to check if the script is being run as the main module
if __name__ == "__main__":
    main()
