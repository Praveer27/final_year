"""
Modern Video Processing Module for Gesture2Speech
Handles video to frame extraction with MediaPipe hand detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import mediapipe as mp
from loguru import logger


class VideoProcessor:
    """
    Advanced video processor with MediaPipe integration
    Extracts frames and detects hand landmarks
    """
    
    def __init__(
        self,
        max_frames: int = 200,
        target_size: Tuple[int, int] = (224, 224),
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize video processor
        
        Args:
            max_frames: Maximum number of frames to extract per video
            target_size: Target frame size (width, height)
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.max_frames = max_frames
        self.target_size = target_size
        self.max_num_hands = max_num_hands
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        logger.info(f"VideoProcessor initialized with max_frames={max_frames}, target_size={target_size}")
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        draw_landmarks: bool = True,
        save_frames: bool = True
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from video with hand detection
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            draw_landmarks: Whether to draw hand landmarks on frames
            save_frames: Whether to save frames to disk
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        if save_frames:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'original_size': (width, height),
            'target_size': self.target_size,
            'extracted_frames': 0,
            'hands_detected': 0
        }
        
        frames = []
        last_valid_frame = None
        frame_count = 0
        
        logger.info(f"Processing video: {video_path.name} (Total frames: {total_frames}, FPS: {fps})")
        
        # Calculate frame skip for uniform sampling
        frame_skip = max(1, total_frames // self.max_frames)
        
        with tqdm(total=min(total_frames, self.max_frames), desc="Extracting frames") as pbar:
            while frame_count < self.max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames for uniform sampling
                if frame_count % frame_skip != 0 and frame_count > 0:
                    continue
                
                # Resize frame
                frame = cv2.resize(frame, self.target_size)
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(frame_rgb)
                
                # Draw landmarks if hands detected
                if results.multi_hand_landmarks:
                    metadata['hands_detected'] += 1
                    
                    if draw_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                    
                    last_valid_frame = frame.copy()
                    frames.append(frame)
                else:
                    # Use last valid frame if no hands detected
                    if last_valid_frame is not None:
                        frames.append(last_valid_frame.copy())
                    else:
                        frames.append(frame)
                
                # Save frame if requested
                if save_frames:
                    frame_filename = output_dir / f"{video_path.stem}_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(str(frame_filename), frames[-1])
                
                frame_count += 1
                pbar.update(1)
        
        # Pad with last frame if needed
        while len(frames) < self.max_frames and last_valid_frame is not None:
            frames.append(last_valid_frame.copy())
            if save_frames:
                frame_filename = output_dir / f"{video_path.stem}_frame_{len(frames)-1:04d}.jpg"
                cv2.imwrite(str(frame_filename), frames[-1])
        
        cap.release()
        
        metadata['extracted_frames'] = len(frames)
        logger.info(f"Extracted {len(frames)} frames from {video_path.name} (Hands detected in {metadata['hands_detected']} frames)")
        
        return frames, metadata
    
    def process_video_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']
    ) -> Dict:
        """
        Process all videos in a directory
        
        Args:
            input_dir: Directory containing videos
            output_dir: Directory to save extracted frames
            video_extensions: List of video file extensions to process
            
        Returns:
            Dictionary with processing statistics
        """
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.rglob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return {'processed': 0, 'failed': 0}
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        stats = {
            'processed': 0,
            'failed': 0,
            'total_frames': 0,
            'videos': []
        }
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                # Create output subdirectory maintaining structure
                relative_path = video_path.relative_to(input_dir)
                video_output_dir = output_dir / relative_path.parent / video_path.stem
                
                frames, metadata = self.extract_frames(
                    video_path,
                    video_output_dir,
                    draw_landmarks=True,
                    save_frames=True
                )
                
                stats['processed'] += 1
                stats['total_frames'] += len(frames)
                stats['videos'].append(metadata)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                stats['failed'] += 1
        
        logger.info(f"Processing complete: {stats['processed']} videos processed, {stats['failed']} failed")
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


if __name__ == "__main__":
    # Test video processor
    from src.utils import setup_logger
    
    setup_logger(log_dir=Path("logs"), log_level="INFO")
    
    processor = VideoProcessor(
        max_frames=200,
        target_size=(224, 224),
        max_num_hands=2
    )
    
    # Example usage
    # processor.process_video_directory(
    #     input_dir=Path("data/videos"),
    #     output_dir=Path("data/frames")
    # )

# Made with Bob
