"""
Advanced Keypoint Extraction Module
Extracts and processes hand keypoints from frames using MediaPipe
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import mediapipe as mp
from loguru import logger
from tqdm import tqdm


class KeypointExtractor:
    """
    Extract hand keypoints from images/frames using MediaPipe
    Supports both single and double hand detection with normalization
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize keypoint extractor
        
        Args:
            max_num_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: Model complexity (0 or 1, higher is more accurate)
        """
        self.max_num_hands = max_num_hands
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        logger.info(f"KeypointExtractor initialized with max_hands={max_num_hands}")
    
    def extract_from_image(
        self,
        image: np.ndarray
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Extract keypoints from a single image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (list of hand keypoints, handedness info)
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        keypoints_list = []
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': getattr(landmark, 'visibility', 1.0)
                    })
                
                hand_info = {
                    'keypoints': keypoints,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score
                }
                keypoints_list.append(hand_info)
        
        return keypoints_list, results
    
    def extract_from_frames(
        self,
        frames_dir: Path,
        output_file: Optional[Path] = None
    ) -> Dict:
        """
        Extract keypoints from all frames in a directory
        
        Args:
            frames_dir: Directory containing frame images
            output_file: Optional path to save keypoints JSON
            
        Returns:
            Dictionary with keypoints data
        """
        if not frames_dir.exists():
            logger.error(f"Frames directory not found: {frames_dir}")
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Get all image files
        image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png")))
        
        if not image_files:
            logger.warning(f"No image files found in {frames_dir}")
            return {}
        
        logger.info(f"Extracting keypoints from {len(image_files)} frames")
        
        all_keypoints = []
        
        for img_path in tqdm(image_files, desc="Extracting keypoints"):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            
            keypoints_list, _ = self.extract_from_image(image)
            
            # Store frame keypoints
            frame_data = {
                'frame_name': img_path.name,
                'hands': keypoints_list
            }
            all_keypoints.append(frame_data)
        
        # Save to JSON if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_keypoints, f, indent=2)
            logger.info(f"Keypoints saved to {output_file}")
        
        return all_keypoints
    
    def normalize_keypoints(
        self,
        keypoints: List[Dict]
    ) -> List[Dict]:
        """
        Normalize keypoints to [0, 1] range
        
        Args:
            keypoints: List of keypoint dictionaries
            
        Returns:
            Normalized keypoints
        """
        if not keypoints:
            return keypoints
        
        normalized = []
        
        for hand_data in keypoints:
            kps = hand_data['keypoints']
            
            # Extract coordinates
            coords = np.array([[kp['x'], kp['y'], kp['z']] for kp in kps])
            
            # Normalize each dimension
            for dim in range(3):
                min_val = coords[:, dim].min()
                max_val = coords[:, dim].max()
                
                if max_val - min_val > 1e-6:
                    coords[:, dim] = (coords[:, dim] - min_val) / (max_val - min_val)
                else:
                    coords[:, dim] = 0.0
            
            # Create normalized keypoints
            normalized_kps = []
            for i, kp in enumerate(kps):
                normalized_kps.append({
                    'x': float(coords[i, 0]),
                    'y': float(coords[i, 1]),
                    'z': float(coords[i, 2]),
                    'visibility': kp['visibility']
                })
            
            normalized.append({
                'keypoints': normalized_kps,
                'handedness': hand_data['handedness'],
                'confidence': hand_data['confidence']
            })
        
        return normalized
    
    def process_dataset(
        self,
        dataset_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Process entire dataset directory structure
        
        Args:
            dataset_dir: Root directory with category/gesture/frames structure
            output_dir: Directory to save extracted keypoints
            
        Returns:
            Processing statistics
        """
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'categories': 0,
            'gestures': 0,
            'total_frames': 0,
            'hands_detected': 0
        }
        
        all_data = {}
        
        # Iterate through categories
        for category_dir in dataset_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            all_data[category_name] = {}
            stats['categories'] += 1
            
            logger.info(f"Processing category: {category_name}")
            
            # Iterate through gestures
            for gesture_dir in category_dir.iterdir():
                if not gesture_dir.is_dir():
                    continue
                
                gesture_name = gesture_dir.name
                stats['gestures'] += 1
                
                logger.info(f"  Processing gesture: {gesture_name}")
                
                # Extract keypoints from frames
                keypoints_data = self.extract_from_frames(gesture_dir)
                
                # Normalize keypoints
                normalized_data = []
                for frame_data in keypoints_data:
                    normalized_hands = self.normalize_keypoints(frame_data['hands'])
                    normalized_data.append({
                        'frame_name': frame_data['frame_name'],
                        'hands': normalized_hands
                    })
                    
                    stats['total_frames'] += 1
                    if normalized_hands:
                        stats['hands_detected'] += len(normalized_hands)
                
                all_data[category_name][gesture_name] = normalized_data
                
                # Save individual gesture keypoints
                gesture_output_dir = output_dir / category_name / gesture_name
                gesture_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = gesture_output_dir / 'keypoints.json'
                with open(output_file, 'w') as f:
                    json.dump(normalized_data, f, indent=2)
        
        # Save complete dataset
        complete_output = output_dir / 'complete_keypoints.json'
        with open(complete_output, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        logger.info(f"Dataset processing complete: {stats}")
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


if __name__ == "__main__":
    # Test keypoint extractor
    from src.utils import setup_logger
    
    setup_logger(log_dir=Path("logs"), log_level="INFO")
    
    extractor = KeypointExtractor(
        max_num_hands=2,
        min_detection_confidence=0.7
    )
    
    # Example usage
    # extractor.process_dataset(
    #     dataset_dir=Path("data/frames"),
    #     output_dir=Path("data/keypoints")
    # )

# Made with Bob
