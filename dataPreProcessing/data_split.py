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
        if len(keypoints_frame) != 0:
            num_points_to_select = int(0.7 * len(keypoints_frame))
            random_points = random.sample(keypoints_frame, num_points_to_select)
        else:
            random_points = [[0.0, 0.0, 0.0]]  # If frame is empty, insert a default point

        if len(random_points) == 0:
            random_points = [[0.0, 0.0, 0.0]]
        output_keypoints_list.append(random_points)

    return output_keypoints_list

def split_data_by_label(keypoints_list, train_ratio=0.8):
    """ Split keypoints list into training and testing sets based on the given ratio. """
    random.shuffle(keypoints_list)
    
    split_idx = int(len(keypoints_list) * train_ratio)
    train_data = keypoints_list[:split_idx]
    test_data = keypoints_list[split_idx:]
    
    return train_data, test_data

def save_split_data(data, output_dir, split_type):
    """ Save the split data into separate directories for training and testing. """
    split_dir = os.path.join(output_dir, split_type)
    os.makedirs(split_dir, exist_ok=True)
    
    for category, labels_data in data.items():
        category_folder = os.path.join(split_dir, category)
        os.makedirs(category_folder, exist_ok=True)

        for label, keypoints_list in labels_data.items():
            label_folder = os.path.join(category_folder, label)
            os.makedirs(label_folder, exist_ok=True)

            output_path = os.path.join(label_folder, 'keypoints.json')
            try:
                with open(output_path, 'w') as outfile:
                    json.dump({label: keypoints_list}, outfile)
                print(f"{split_type.capitalize()} data saved: {output_path}")
            except Exception as e:
                print(f"Error writing {split_type} data to {output_path}: {e}")

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

    # Extract the directory path of the input file
    input_dir = os.path.dirname(input_file)

    # Set the folder name to 'split_data' in the same directory as the input file
    base_folder = os.path.join(input_dir, 'split_data')
    os.makedirs(base_folder, exist_ok=True)
    print(f"Main folder created: {base_folder}")

    # Process and split data into train and test sets
    train_data = {}
    test_data = {}
    
    for category, labels_data in keypoints_data.items():
        train_data[category] = {}
        test_data[category] = {}
        
        for label, keypoints_list in labels_data.items():
            # Process keypoints for each label
            processed_keypoints_list = process_keypoints(keypoints_list)

            # Split processed keypoints into 80% train and 20% test
            train_keypoints, test_keypoints = split_data_by_label(processed_keypoints_list, train_ratio=0.8)
            
            # Add split data to respective train and test dictionaries
            train_data[category][label] = train_keypoints
            test_data[category][label] = test_keypoints

    # Save training data
    save_split_data(train_data, base_folder, split_type='train')

    # Save testing data
    save_split_data(test_data, base_folder, split_type='test')

# Corrected condition to check if the script is being run as the main module
if __name__ == "__main__":
    main()
