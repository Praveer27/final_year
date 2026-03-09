"""
Main Entry Point for Gesture2Speech Project
Provides CLI interface for all operations
"""

import argparse
import sys
from pathlib import Path
import torch
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger, config
from src.preprocessing import VideoProcessor, KeypointExtractor, KeypointAugmenter
from src.models.gesture_model import create_model


def setup_directories():
    """Create all necessary directories"""
    logger.info("Setting up project directories...")
    config.create_directories()
    logger.info("Directories created successfully")


def process_videos(args):
    """Process videos to extract frames"""
    logger.info("Starting video processing...")
    
    processor = VideoProcessor(
        max_frames=config.get('video_processing.max_frames', 200),
        target_size=tuple(config.get('video_processing.frame_size', [224, 224])),
        max_num_hands=config.get('mediapipe.max_num_hands', 2),
        min_detection_confidence=config.get('mediapipe.min_detection_confidence', 0.7)
    )
    
    input_dir = Path(args.input) if args.input else config.get_paths()['videos']
    output_dir = Path(args.output) if args.output else config.get_paths()['frames']
    
    stats = processor.process_video_directory(input_dir, output_dir)
    
    logger.info(f"Video processing complete: {stats}")


def extract_keypoints(args):
    """Extract keypoints from frames"""
    logger.info("Starting keypoint extraction...")
    
    extractor = KeypointExtractor(
        max_num_hands=config.get('mediapipe.max_num_hands', 2),
        min_detection_confidence=config.get('mediapipe.min_detection_confidence', 0.7),
        model_complexity=config.get('mediapipe.model_complexity', 1)
    )
    
    input_dir = Path(args.input) if args.input else config.get_paths()['frames']
    output_dir = Path(args.output) if args.output else config.get_paths()['keypoints']
    
    stats = extractor.process_dataset(input_dir, output_dir)
    
    logger.info(f"Keypoint extraction complete: {stats}")


def augment_data(args):
    """Augment keypoint data"""
    logger.info("Starting data augmentation...")
    
    augmenter = KeypointAugmenter(
        rotation_range=config.get('augmentation.rotation_range', 25.0),
        scale_range=tuple(config.get('augmentation.zoom_range', [0.8, 1.2])),
        augmentation_factor=config.get('augmentation.augmentation_factor', 3)
    )
    
    input_file = Path(args.input) if args.input else config.get_paths()['keypoints'] / 'complete_keypoints.json'
    output_file = Path(args.output) if args.output else config.get_paths()['keypoints'] / 'augmented_keypoints.json'
    
    stats = augmenter.augment_dataset(input_file, output_file)
    
    logger.info(f"Data augmentation complete: {stats}")


def train_model(args):
    """Train gesture recognition model"""
    logger.info("Starting model training...")
    
    # This is a placeholder - full implementation would require dataset class
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('performance.use_gpu', True) else 'cpu')
    
    model = create_model(
        model_type=config.get('model.architecture', 'lstm'),
        input_size=63,  # 21 keypoints * 3 coordinates
        num_classes=config.get_num_classes(),
        hidden_size=config.get('model.lstm.hidden_size', 256),
        num_layers=config.get('model.lstm.num_layers', 3),
        dropout=config.get('model.lstm.dropout', 0.3),
        bidirectional=config.get('model.lstm.bidirectional', True)
    )
    
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training on device: {device}")
    
    # TODO: Implement full training pipeline with dataset
    logger.warning("Full training pipeline not yet implemented. Model architecture created successfully.")


def run_inference(args):
    """Run inference on video or webcam"""
    logger.info("Starting inference...")
    
    # TODO: Implement inference pipeline
    logger.warning("Inference pipeline not yet implemented")


def run_pipeline(args):
    """Run complete pipeline from videos to trained model"""
    logger.info("Running complete pipeline...")
    
    # Setup
    setup_directories()
    
    # Process videos
    if not args.skip_videos:
        logger.info("Step 1/4: Processing videos...")
        process_videos(args)
    
    # Extract keypoints
    if not args.skip_keypoints:
        logger.info("Step 2/4: Extracting keypoints...")
        extract_keypoints(args)
    
    # Augment data
    if not args.skip_augmentation:
        logger.info("Step 3/4: Augmenting data...")
        augment_data(args)
    
    # Train model
    if not args.skip_training:
        logger.info("Step 4/4: Training model...")
        train_model(args)
    
    logger.info("Pipeline complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Gesture2Speech - Sign Language to Speech Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py pipeline
  
  # Process videos only
  python main.py process-videos --input data/videos --output data/frames
  
  # Extract keypoints
  python main.py extract-keypoints --input data/frames --output data/keypoints
  
  # Augment data
  python main.py augment-data
  
  # Train model
  python main.py train
  
  # Run inference
  python main.py inference --input video.mp4
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup project directories')
    
    # Process videos command
    process_parser = subparsers.add_parser('process-videos', help='Process videos to extract frames')
    process_parser.add_argument('--input', type=str, help='Input videos directory')
    process_parser.add_argument('--output', type=str, help='Output frames directory')
    
    # Extract keypoints command
    extract_parser = subparsers.add_parser('extract-keypoints', help='Extract keypoints from frames')
    extract_parser.add_argument('--input', type=str, help='Input frames directory')
    extract_parser.add_argument('--output', type=str, help='Output keypoints directory')
    
    # Augment data command
    augment_parser = subparsers.add_parser('augment-data', help='Augment keypoint data')
    augment_parser.add_argument('--input', type=str, help='Input keypoints file')
    augment_parser.add_argument('--output', type=str, help='Output augmented file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train gesture recognition model')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--input', type=str, help='Input video file or "webcam"')
    inference_parser.add_argument('--model', type=str, help='Path to trained model')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--skip-videos', action='store_true', help='Skip video processing')
    pipeline_parser.add_argument('--skip-keypoints', action='store_true', help='Skip keypoint extraction')
    pipeline_parser.add_argument('--skip-augmentation', action='store_true', help='Skip data augmentation')
    pipeline_parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = config.get_paths().get('logs', Path('logs'))
    setup_logger(log_dir=log_dir, log_level=args.log_level)
    
    logger.info("=" * 80)
    logger.info(f"Gesture2Speech v{config.get('project.version')}")
    logger.info(f"Command: {args.command}")
    logger.info("=" * 80)
    
    # Execute command
    if args.command == 'setup':
        setup_directories()
    elif args.command == 'process-videos':
        process_videos(args)
    elif args.command == 'extract-keypoints':
        extract_keypoints(args)
    elif args.command == 'augment-data':
        augment_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'inference':
        run_inference(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Execution completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
