"""
Advanced Data Augmentation Module
Implements various augmentation techniques for gesture recognition
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random
from loguru import logger
from tqdm import tqdm


class KeypointAugmenter:
    """
    Augment keypoint data for gesture recognition
    Includes rotation, scaling, translation, and noise injection
    """
    
    def __init__(
        self,
        rotation_range: float = 25.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_range: float = 0.1,
        noise_std: float = 0.02,
        augmentation_factor: int = 3
    ):
        """
        Initialize augmenter
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            scale_range: Min and max scale factors
            translation_range: Maximum translation as fraction of range
            noise_std: Standard deviation for Gaussian noise
            augmentation_factor: Number of augmented samples per original
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.augmentation_factor = augmentation_factor
        
        logger.info(f"KeypointAugmenter initialized with factor={augmentation_factor}")
    
    def rotate_keypoints(
        self,
        keypoints: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """
        Rotate keypoints around center
        
        Args:
            keypoints: Array of shape (N, 3) with x, y, z coordinates
            angle: Rotation angle in degrees
            
        Returns:
            Rotated keypoints
        """
        # Convert to radians
        theta = np.radians(angle)
        
        # Rotation matrix for XY plane
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # Center keypoints
        center = keypoints.mean(axis=0)
        centered = keypoints - center
        
        # Apply rotation
        rotated = centered @ rotation_matrix.T
        
        # Restore center
        return rotated + center
    
    def scale_keypoints(
        self,
        keypoints: np.ndarray,
        scale_factor: float
    ) -> np.ndarray:
        """
        Scale keypoints around center
        
        Args:
            keypoints: Array of shape (N, 3)
            scale_factor: Scaling factor
            
        Returns:
            Scaled keypoints
        """
        center = keypoints.mean(axis=0)
        centered = keypoints - center
        scaled = centered * scale_factor
        return scaled + center
    
    def translate_keypoints(
        self,
        keypoints: np.ndarray,
        translation: np.ndarray
    ) -> np.ndarray:
        """
        Translate keypoints
        
        Args:
            keypoints: Array of shape (N, 3)
            translation: Translation vector
            
        Returns:
            Translated keypoints
        """
        return keypoints + translation
    
    def add_noise(
        self,
        keypoints: np.ndarray,
        noise_std: float
    ) -> np.ndarray:
        """
        Add Gaussian noise to keypoints
        
        Args:
            keypoints: Array of shape (N, 3)
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy keypoints
        """
        noise = np.random.normal(0, noise_std, keypoints.shape)
        return keypoints + noise
    
    def augment_single_frame(
        self,
        keypoints: np.ndarray
    ) -> np.ndarray:
        """
        Apply random augmentation to a single frame
        
        Args:
            keypoints: Array of shape (N, 3)
            
        Returns:
            Augmented keypoints
        """
        augmented = keypoints.copy()
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            augmented = self.rotate_keypoints(augmented, angle)
        
        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            augmented = self.scale_keypoints(augmented, scale)
        
        # Random translation
        if random.random() > 0.5:
            translation = np.random.uniform(
                -self.translation_range,
                self.translation_range,
                size=3
            )
            augmented = self.translate_keypoints(augmented, translation)
        
        # Add noise
        if random.random() > 0.5:
            augmented = self.add_noise(augmented, self.noise_std)
        
        # Clip to valid range [0, 1]
        augmented = np.clip(augmented, 0.0, 1.0)
        
        return augmented
    
    def augment_sequence(
        self,
        keypoints_sequence: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Augment a sequence of keypoints (video frames)
        
        Args:
            keypoints_sequence: List of keypoint arrays
            
        Returns:
            Augmented sequence
        """
        # Use same augmentation parameters for entire sequence
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        scale = random.uniform(*self.scale_range)
        translation = np.random.uniform(
            -self.translation_range,
            self.translation_range,
            size=3
        )
        
        augmented_sequence = []
        
        for keypoints in keypoints_sequence:
            augmented = keypoints.copy()
            
            # Apply consistent transformations
            augmented = self.rotate_keypoints(augmented, angle)
            augmented = self.scale_keypoints(augmented, scale)
            augmented = self.translate_keypoints(augmented, translation)
            
            # Add frame-specific noise
            augmented = self.add_noise(augmented, self.noise_std)
            
            # Clip to valid range
            augmented = np.clip(augmented, 0.0, 1.0)
            
            augmented_sequence.append(augmented)
        
        return augmented_sequence
    
    def augment_dataset(
        self,
        input_file: Path,
        output_file: Path
    ) -> Dict:
        """
        Augment entire dataset from JSON file
        
        Args:
            input_file: Path to input keypoints JSON
            output_file: Path to save augmented data
            
        Returns:
            Augmentation statistics
        """
        logger.info(f"Loading keypoints from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        augmented_data = {}
        stats = {
            'original_samples': 0,
            'augmented_samples': 0,
            'total_samples': 0
        }
        
        # Process each category
        for category, gestures in tqdm(data.items(), desc="Augmenting categories"):
            augmented_data[category] = {}
            
            for gesture, frames_data in gestures.items():
                augmented_data[category][gesture] = []
                
                # Add original data
                augmented_data[category][gesture].extend(frames_data)
                stats['original_samples'] += len(frames_data)
                
                # Generate augmented samples
                for _ in range(self.augmentation_factor):
                    for frame_data in frames_data:
                        augmented_frame = frame_data.copy()
                        
                        # Augment each hand
                        if 'hands' in frame_data:
                            augmented_hands = []
                            
                            for hand_data in frame_data['hands']:
                                # Extract keypoints
                                kps = hand_data['keypoints']
                                coords = np.array([[kp['x'], kp['y'], kp['z']] for kp in kps])
                                
                                # Augment
                                aug_coords = self.augment_single_frame(coords)
                                
                                # Convert back to dict format
                                aug_kps = []
                                for i, kp in enumerate(kps):
                                    aug_kps.append({
                                        'x': float(aug_coords[i, 0]),
                                        'y': float(aug_coords[i, 1]),
                                        'z': float(aug_coords[i, 2]),
                                        'visibility': kp.get('visibility', 1.0)
                                    })
                                
                                augmented_hands.append({
                                    'keypoints': aug_kps,
                                    'handedness': hand_data['handedness'],
                                    'confidence': hand_data['confidence']
                                })
                            
                            augmented_frame['hands'] = augmented_hands
                        
                        augmented_data[category][gesture].append(augmented_frame)
                        stats['augmented_samples'] += 1
        
        stats['total_samples'] = stats['original_samples'] + stats['augmented_samples']
        
        # Save augmented data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        logger.info(f"Augmentation complete: {stats}")
        logger.info(f"Augmented data saved to {output_file}")
        
        return stats


if __name__ == "__main__":
    # Test augmenter
    from src.utils import setup_logger
    
    setup_logger(log_dir=Path("logs"), log_level="INFO")
    
    augmenter = KeypointAugmenter(
        rotation_range=25.0,
        scale_range=(0.8, 1.2),
        augmentation_factor=3
    )
    
    # Example usage
    # augmenter.augment_dataset(
    #     input_file=Path("data/keypoints/complete_keypoints.json"),
    #     output_file=Path("data/keypoints/augmented_keypoints.json")
    # )

# Made with Bob
