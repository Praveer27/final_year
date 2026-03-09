import cv2
import os
from os.path import join, exists
import mediapipe as mp
from tqdm import tqdm

hc = []  # For tracking frame info

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

def convert(dataset_folder, target_folder):
    rootPath = os.getcwd()  # Save current working directory
    majorData = os.path.abspath(target_folder)  # Target folder to save frames
    
    if not exists(majorData):
        os.makedirs(majorData)

    dataset_folder = os.path.abspath(dataset_folder)

    print(f"Source Directory containing gesture categories: {dataset_folder}")
    print(f"Destination Directory for frames: {majorData}\n")

    # Iterate over categories (pronouns, adjectives, places, etc.)
    for category in tqdm(os.listdir(dataset_folder), unit='categories', ascii=True):
        category_path = join(dataset_folder, category)  # Path to the category folder
        os.chdir(category_path)

        # Iterate over gestures in each category
        for gesture in tqdm(os.listdir(category_path), unit='gestures', ascii=True):
            gesture_path = join(category_path, gesture)  # Path to gesture folder
            os.chdir(gesture_path)

            gesture_frames_path = join(majorData, category, gesture)  # Path to save frames
            if not os.path.exists(gesture_frames_path):
                os.makedirs(gesture_frames_path)

            videos = [video for video in os.listdir(gesture_path) if os.path.isfile(video)]

            # Process each video
            for video in tqdm(videos, unit='videos', ascii=True):
                video_path = os.path.abspath(video)
                cap = cv2.VideoCapture(video_path)  # Capture video
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps != 0:
                    duration = frameCount / fps
                else:
                    duration = 0
                lastFrame = None
                count = 0  # Frame counter

                print(f"Processing video: {video_path} (Total Frames: {frameCount})")

                os.chdir(gesture_frames_path)

                while count < 200:
                    ret, frame = cap.read()  # Read frame
                    if not ret:
                        break

                    framename = os.path.splitext(video)[0] + f"_frame_{count}.jpeg"
                    hc.append([join(gesture_frames_path, framename), category, gesture, frameCount])

                    # Use Mediapipe Hands to detect keypoints
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        # Draw hand landmarks on the frame for visualization (optional)
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        lastFrame = frame
                        cv2.imwrite(framename, frame)  # Save frame with keypoints drawn
                    else:
                        if lastFrame is not None:
                            cv2.imwrite(framename, lastFrame)  # Save last valid frame if no hands are detected

                    count += 1

                    if count >= 200:
                        while count < frameCount:
                            cv2.imwrite(framename, lastFrame)
                            count += 1
                        break

                cap.release()  # Release video capture
                cv2.destroyAllWindows()  # Close OpenCV windows

            os.chdir(category_path)  # Go back to category folder

        os.chdir(dataset_folder)  # Go back to dataset folder

    os.chdir(rootPath)  # Return to root path

if __name__ == '__main__':
    dataset_folder = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\videos'
    target_folder = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames'
    convert(dataset_folder, target_folder) # Folder where extracted frames will be saved
    
    '''code creates 200 frames but with repeating static frames last repeated'''
# import cv2
# import os
# from os.path import join, exists
# import mediapipe as mp
# from tqdm import tqdm

# hc = []  # For tracking frame info

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# def convert(dataset_folder, target_folder):
#     rootPath = os.getcwd()  # Save current working directory
#     majorData = os.path.abspath(target_folder)  # Target folder to save frames
    
#     if not exists(majorData):
#         os.makedirs(majorData)

#     dataset_folder = os.path.abspath(dataset_folder)

#     print(f"Source Directory containing gesture categories: {dataset_folder}")
#     print(f"Destination Directory for frames: {majorData}\n")

#     # Iterate over categories (pronouns, adjectives, places, etc.)
#     for category in tqdm(os.listdir(dataset_folder), unit='categories', ascii=True):
#         category_path = join(dataset_folder, category)  # Path to the category folder
#         os.chdir(category_path)

#         # Iterate over gestures in each category
#         for gesture in tqdm(os.listdir(category_path), unit='gestures', ascii=True):
#             gesture_path = join(category_path, gesture)  # Path to gesture folder
#             os.chdir(gesture_path)

#             gesture_frames_path = join(majorData, category, gesture)  # Path to save frames
#             if not os.path.exists(gesture_frames_path):
#                 os.makedirs(gesture_frames_path)

#             videos = [video for video in os.listdir(gesture_path) if os.path.isfile(video)]

#             # Process each video
#             for video in tqdm(videos, unit='videos', ascii=True):
#                 video_path = os.path.abspath(video)
#                 cap = cv2.VideoCapture(video_path)  # Capture video
#                 frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 lastFrame = None
#                 count = 0  # Frame counter

#                 print(f"Processing video: {video_path} (Total Frames: {frameCount})")

#                 os.chdir(gesture_frames_path)

#                 # Extract up to 200 frames per video
#                 while count < 201:
#                     ret, frame = cap.read()  # Read frame
#                     if not ret:
#                         break

#                     framename = os.path.splitext(video)[0] + f"_frame_{count}.jpeg"
#                     hc.append([join(gesture_frames_path, framename), category, gesture, frameCount])

#                     # Use Mediapipe Hands to detect keypoints
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
#                     results = hands.process(frame_rgb)

#                     if results.multi_hand_landmarks:
#                         # Draw hand landmarks on the frame for visualization (optional)
#                         for hand_landmarks in results.multi_hand_landmarks:
#                             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                         lastFrame = frame
#                         cv2.imwrite(framename, frame)  # Save frame with keypoints drawn
#                     else:
#                         if lastFrame is not None:
#                             cv2.imwrite(framename, lastFrame)  # Save last valid frame if no hands are detected

#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                     count += 1

#                 # If video has less than 200 frames, repeat last frame
#                 while count < 201:
#                     framename = os.path.splitext(video)[0] + f"_frame_{count}.jpeg"
#                     hc.append([join(gesture_frames_path, framename), category, gesture, frameCount])
#                     if lastFrame is not None and not os.path.exists(framename):
#                         cv2.imwrite(framename, lastFrame)
#                     count += 1

#                 cap.release()  # Release video capture
#                 cv2.destroyAllWindows()  # Close OpenCV windows

#             os.chdir(category_path)  # Go back to category folder

#         os.chdir(dataset_folder)  # Go back to dataset folder

#     os.chdir(rootPath)  # Return to root path

# if __name__ == '__main__':
#     dataset_folder = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\videos'  # Folder containing gesture videos organized by category
#     target_folder = r'D:\FINAL_YEAR_PROJECT\Gesture2Speech\frames'  # Folder where extracted frames will be saved
#     convert(dataset_folder, target_folder)
