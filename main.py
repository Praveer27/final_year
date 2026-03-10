"""
Main Entry Point for Gesture2Speech Project
Provides CLI interface for all operations
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

# Add project root to path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger, config
from src.preprocessing import VideoProcessor, KeypointExtractor, KeypointAugmenter
from src.models.gesture_model import create_model
from src.training.keypoint_dataset import GestureSequenceDataset


def setup_directories():
    """Create all necessary directories"""
    logger.info("Setting up project directories...")
    config.create_directories()
    logger.info("Directories created successfully")


def process_videos(args):
    """Process videos to extract frames"""
    logger.info("Starting video processing...")

    processor = VideoProcessor(
        max_frames=config.get("video_processing.max_frames", 200),
        target_size=tuple(config.get("video_processing.frame_size", [224, 224])),
        max_num_hands=config.get("mediapipe.max_num_hands", 2),
        min_detection_confidence=config.get("mediapipe.min_detection_confidence", 0.7),
    )

    input_arg = getattr(args, "input", None)
    output_arg = getattr(args, "output", None)

    input_dir = Path(input_arg) if input_arg else Path(config.get_paths()["videos"])
    output_dir = Path(output_arg) if output_arg else Path(config.get_paths()["frames"])

    stats = processor.process_video_directory(input_dir, output_dir)
    logger.info(f"Video processing complete: {stats}")


def extract_keypoints(args):
    """Extract keypoints from frames"""
    logger.info("Starting keypoint extraction...")

    extractor = KeypointExtractor(
        max_num_hands=config.get("mediapipe.max_num_hands", 2),
        min_detection_confidence=config.get("mediapipe.min_detection_confidence", 0.7),
        min_tracking_confidence=config.get("mediapipe.min_tracking_confidence", 0.5),
        model_complexity=config.get("mediapipe.model_complexity", 1),
        static_image_mode=config.get("mediapipe.static_image_mode", False),
    )

    input_arg = getattr(args, "input", None)
    output_arg = getattr(args, "output", None)

    input_dir = Path(input_arg) if input_arg else Path(config.get_paths()["frames"])
    output_dir = Path(output_arg) if output_arg else Path(config.get_paths()["keypoints"])

    stats = extractor.process_dataset(input_dir, output_dir)
    logger.info(f"Keypoint extraction complete: {stats}")


def augment_data(args):
    """Augment keypoint data"""
    logger.info("Starting data augmentation...")

    augmenter = KeypointAugmenter(
        rotation_range=config.get("augmentation.rotation_range", 25.0),
        scale_range=tuple(config.get("augmentation.zoom_range", [0.8, 1.2])),
        augmentation_factor=config.get("augmentation.augmentation_factor", 3),
    )

    input_arg = getattr(args, "input", None)
    output_arg = getattr(args, "output", None)

    input_file = Path(input_arg) if input_arg else Path(config.get_paths()["keypoints"]) / "complete_keypoints.json"
    output_file = Path(output_arg) if output_arg else Path(config.get_paths()["keypoints"]) / "augmented_keypoints.json"

    stats = augmenter.augment_dataset(input_file, output_file)
    logger.info(f"Data augmentation complete: {stats}")


def train_model(args):
    """Train gesture recognition model (PyTorch)"""
    logger.info("Starting model training...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.get("performance.use_gpu", True) else "cpu"
    )

    keypoints_file = Path(config.get_paths()["keypoints"]) / "augmented_keypoints.json"
    seq_len = config.get("model.sequence_length", 45)

    ds = GestureSequenceDataset(str(keypoints_file), seq_len=seq_len)
    if len(ds) == 0:
        raise RuntimeError(
            f"No training samples found in {keypoints_file}. "
            "Run extract-keypoints and augment-data first."
        )

    num_classes = len(ds.label_map)
    logger.info(f"Dataset loaded: samples={len(ds)} classes={num_classes}")
    logger.info(f"Label map: {ds.label_map}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = create_model(
        model_type=config.get("model.architecture", "lstm"),
        input_size=63,  # 21 keypoints * 3 coords
        num_classes=num_classes,
        hidden_size=config.get("model.lstm.hidden_size", 256),
        num_layers=config.get("model.lstm.num_layers", 3),
        dropout=config.get("model.lstm.dropout", 0.3),
        bidirectional=config.get("model.lstm.bidirectional", True),
    ).to(device)

    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training on device: {device}")

    criterion = nn.CrossEntropyLoss()
    lr = float(config.get("training.learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dl:
            xb = xb.to(device)  # (B, T, 63)
            yb = yb.to(device)  # (B,)

            optimizer.zero_grad()
            logits = model(xb)  # expected (B, num_classes)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        logger.info(f"Epoch {epoch}/{args.epochs} - loss={avg_loss:.4f} acc={acc:.4f}")

    ckpt_dir = Path(config.get_paths().get("checkpoints", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / "gesture_model.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "label_map": ds.label_map,
            "seq_len": seq_len,
            "input_size": 63,
        },
        out_path,
    )
    logger.info(f"Saved model checkpoint: {out_path}")



def run_inference(args):
    """Run inference on keypoints (test) or webcam (live)"""
    logger.info("Starting inference...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.get("performance.use_gpu", True) else "cpu"
    )

    ckpt_path = args.model or str(Path(config.get_paths().get("checkpoints", "models/checkpoints")) / "gesture_model.pt")

    from src.inference.sequence_infer import load_checkpoint, predict_sequence

    # Load ckpt first to get num_classes
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = len(ckpt["label_map"])

    model_kwargs = dict(
        model_type=config.get("model.architecture", "lstm"),
        input_size=63,
        num_classes=num_classes,
        hidden_size=config.get("model.lstm.hidden_size", 256),
        num_layers=config.get("model.lstm.num_layers", 3),
        dropout=config.get("model.lstm.dropout", 0.3),
        bidirectional=config.get("model.lstm.bidirectional", True),
    )

    model, id_to_label, seq_len, input_size = load_checkpoint(
        ckpt_path=ckpt_path,
        create_model_fn=create_model,
        model_kwargs=model_kwargs,
        device=device,
    )

    # --- Mode 1: quick offline test ---
    if args.input == "keypoints":
        from src.training.keypoint_dataset import GestureSequenceDataset
        ds = GestureSequenceDataset("data/keypoints/augmented_keypoints.json", seq_len=seq_len)
        x, y = ds[0]
        probs = predict_sequence(model, x.numpy(), device=device)
        pred_id = int(probs.argmax())
        logger.info(f"Predicted: {id_to_label[pred_id]} (id={pred_id})")
        logger.info(f"Actual label id: {int(y.item())}")
        return

    # --- Mode 2: webcam live inference ---
    if args.input != "webcam":
        logger.warning('Supported inputs: "keypoints" or "webcam" (for now).')
        return

    import cv2
    import numpy as np

    # MediaPipe Hands (same params as extraction)
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.get("mediapipe.max_num_hands", 2),
        min_detection_confidence=config.get("mediapipe.min_detection_confidence", 0.3),
        min_tracking_confidence=config.get("mediapipe.min_tracking_confidence", 0.3),
        model_complexity=config.get("mediapipe.model_complexity", 1),
    )

    def hand_to_vec(results) -> np.ndarray:
        """Return (63,) from first detected hand; else zeros."""
        if not results.multi_hand_landmarks:
            return np.zeros((63,), dtype=np.float32)
        lm = results.multi_hand_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)
        return arr.reshape(-1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permissions for Terminal/VS Code.")

    buffer = []  # list of (63,) vectors
    print_every_n_frames = 30  # ~1 sec if ~30fps

    logger.info("Webcam inference started. Press 'q' to quit.")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # optional: mirror for user-friendly view
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            vec = hand_to_vec(results)
            buffer.append(vec)

            # keep rolling window of seq_len
            if len(buffer) > seq_len:
                buffer = buffer[-seq_len:]

            # predict only when buffer full
            if len(buffer) == seq_len and (frame_count % print_every_n_frames == 0):
                seq = np.stack(buffer, axis=0)  # (T,63)
                probs = predict_sequence(model, seq, device=device)
                pred_id = int(probs.argmax())
                pred_label = id_to_label[pred_id]
                conf = float(probs[pred_id])
                logger.info(f"Predicted: {pred_label} (conf={conf:.3f})")

            # show preview
            cv2.imshow("Gesture2Speech - Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass

def run_pipeline(args):
    """Run complete pipeline from videos to trained model"""
    logger.info("Running complete pipeline...")

    setup_directories()

    if not args.skip_videos:
        logger.info("Step 1/4: Processing videos...")
        process_videos(args)

    if not args.skip_keypoints:
        logger.info("Step 2/4: Extracting keypoints...")
        extract_keypoints(args)

    if not args.skip_augmentation:
        logger.info("Step 3/4: Augmenting data...")
        augment_data(args)

    if not args.skip_training:
        logger.info("Step 4/4: Training model...")
        train_model(args)

    logger.info("Pipeline complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Gesture2Speech - Sign Language to Speech Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pipeline
  python main.py process-videos --input data/videos --output data/frames
  python main.py extract-keypoints --input data/frames --output data/keypoints
  python main.py augment-data
  python main.py train --epochs 5 --batch-size 8
  python main.py inference --input webcam
        """,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # setup
    subparsers.add_parser("setup", help="Setup project directories")

    # process-videos
    process_parser = subparsers.add_parser("process-videos", help="Process videos to extract frames")
    process_parser.add_argument("--input", type=str, help="Input videos directory")
    process_parser.add_argument("--output", type=str, help="Output frames directory")

    # extract-keypoints
    extract_parser = subparsers.add_parser("extract-keypoints", help="Extract keypoints from frames")
    extract_parser.add_argument("--input", type=str, help="Input frames directory")
    extract_parser.add_argument("--output", type=str, help="Output keypoints directory")

    # augment-data
    augment_parser = subparsers.add_parser("augment-data", help="Augment keypoint data")
    augment_parser.add_argument("--input", type=str, help="Input keypoints file")
    augment_parser.add_argument("--output", type=str, help="Output augmented file")

    # train
    train_parser = subparsers.add_parser("train", help="Train gesture recognition model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # inference
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument('--input', type=str, help='Input video file or "webcam"')
    inference_parser.add_argument("--model", type=str, help="Path to trained model")

    # pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--skip-videos", action="store_true", help="Skip video processing")
    pipeline_parser.add_argument("--skip-keypoints", action="store_true", help="Skip keypoint extraction")
    pipeline_parser.add_argument("--skip-augmentation", action="store_true", help="Skip data augmentation")
    pipeline_parser.add_argument("--skip-training", action="store_true", help="Skip model training")

    args = parser.parse_args()

    log_dir = config.get_paths().get("logs", Path("logs"))
    setup_logger(log_dir=log_dir, log_level=args.log_level)

    logger.info("=" * 80)
    logger.info(f"Gesture2Speech v{config.get('project.version')}")
    logger.info(f"Command: {args.command}")
    logger.info("=" * 80)

    if args.command == "setup":
        setup_directories()
    elif args.command == "process-videos":
        process_videos(args)
    elif args.command == "extract-keypoints":
        extract_keypoints(args)
    elif args.command == "augment-data":
        augment_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "inference":
        run_inference(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Execution completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()