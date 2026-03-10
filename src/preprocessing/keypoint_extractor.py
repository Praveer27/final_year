"""
Advanced Keypoint Extraction Module
Extracts and processes hand keypoints from frames using MediaPipe
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


class KeypointExtractor:
    """
    Extract hand keypoints from images/frames using MediaPipe.

    Outputs per-frame records containing:
      - video_id (parent folder name under gesture directory)
      - frame_idx (parsed from filename pattern *_frame_0001)
      - frame_path (relative path under the gesture directory)
      - hands (list of detected hands)
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
    ):
        self.max_num_hands = max_num_hands
        self.hands = None

        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=static_image_mode,  # IMPORTANT: False for video sequences
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    model_complexity=model_complexity,
                )
                logger.info(
                    "KeypointExtractor initialized with MediaPipe solutions API "
                    f"(static_image_mode={static_image_mode}, max_hands={max_num_hands}, "
                    f"min_det={min_detection_confidence}, min_track={min_tracking_confidence})"
                )
            else:
                logger.warning("MediaPipe installed but 'solutions' API not available.")
                logger.warning("Keypoint extraction will return empty keypoints.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            logger.warning("Keypoint extraction will return empty keypoints.")

    def extract_from_image(self, image: np.ndarray) -> Tuple[List[Dict], Optional[object]]:
        """
        Extract keypoints from a single image (BGR).
        Returns: (hands_list, raw_results)
        """
        if self.hands is None:
            return [], None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        keypoints_list: List[Dict] = []
        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append(
                        {
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(getattr(lm, "visibility", 1.0)),
                        }
                    )

                keypoints_list.append(
                    {
                        "keypoints": keypoints,
                        "handedness": handedness.classification[0].label,
                        "confidence": float(handedness.classification[0].score),
                    }
                )

        return keypoints_list, results

    @staticmethod
    def _parse_frame_idx(img_path: Path) -> int:
        """
        Extract frame index from filename like: MVI_5177_frame_0007.jpg
        Returns -1 if not found.
        """
        m = re.search(r"_frame_(\d+)", img_path.stem)
        return int(m.group(1)) if m else -1

    def extract_from_frames(self, frames_dir: Path, output_file: Optional[Path] = None) -> List[Dict]:
        """
        Extract keypoints from all frames inside a gesture directory.

        NOTE: Your frames are stored like:
          data/frames/<Category>/<Gesture>/<VideoID>/<frame>.jpg
        So we use rglob to find all frames under frames_dir recursively.
        """
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        image_files = sorted(
            list(frames_dir.rglob("*.jpg"))
            + list(frames_dir.rglob("*.jpeg"))
            + list(frames_dir.rglob("*.png"))
        )

        if not image_files:
            logger.warning(f"No image files found in {frames_dir}")
            return []

        logger.info(f"Extracting keypoints from {len(image_files)} frames under {frames_dir}")

        all_keypoints: List[Dict] = []
        for img_path in tqdm(image_files, desc=f"Extracting keypoints ({frames_dir.name})"):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            hands_list, _ = self.extract_from_image(image)

            # video_id is the immediate parent folder (e.g., "MVI_5177")
            video_id = img_path.parent.name
            frame_idx = self._parse_frame_idx(img_path)

            frame_data = {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "frame_path": str(img_path.relative_to(frames_dir)),
                "hands": hands_list,
            }
            all_keypoints.append(frame_data)

        # Ensure stable ordering for sequence building later
        all_keypoints.sort(key=lambda x: (x.get("video_id", ""), x.get("frame_idx", -1), x.get("frame_path", "")))

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(all_keypoints, f, indent=2)
            logger.info(f"Keypoints saved to {output_file}")

        return all_keypoints

    def normalize_keypoints(self, hands: List[Dict]) -> List[Dict]:
        """
        Normalize keypoints for each detected hand to [0, 1] per-dimension.
        If keypoints are missing/malformed, returns them unchanged or empty.
        """
        if not hands:
            return []

        normalized_hands: List[Dict] = []
        for hand_data in hands:
            kps = hand_data.get("keypoints", [])
            if not kps:
                continue

            coords = np.array([[kp["x"], kp["y"], kp["z"]] for kp in kps], dtype=np.float32)  # (21,3)

            # Normalize x,y,z independently
            for dim in range(3):
                min_val = float(coords[:, dim].min())
                max_val = float(coords[:, dim].max())
                if max_val - min_val > 1e-6:
                    coords[:, dim] = (coords[:, dim] - min_val) / (max_val - min_val)
                else:
                    coords[:, dim] = 0.0

            normalized_kps = []
            for i, kp in enumerate(kps):
                normalized_kps.append(
                    {
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                        "z": float(coords[i, 2]),
                        "visibility": float(kp.get("visibility", 1.0)),
                    }
                )

            normalized_hands.append(
                {
                    "keypoints": normalized_kps,
                    "handedness": hand_data.get("handedness"),
                    "confidence": float(hand_data.get("confidence", 0.0)),
                }
            )

        return normalized_hands

    def process_dataset(self, dataset_dir: Path, output_dir: Path) -> Dict:
        """
        Process entire dataset directory structure:
          dataset_dir/<Category>/<Gesture>/...frames...

        Saves:
          output_dir/<Category>/<Gesture>/keypoints.json
          output_dir/complete_keypoints.json
        """
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"categories": 0, "gestures": 0, "total_frames": 0, "hands_detected": 0}
        all_data: Dict[str, Dict[str, List[Dict]]] = {}

        for category_dir in dataset_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            all_data[category_name] = {}
            stats["categories"] += 1
            logger.info(f"Processing category: {category_name}")

            for gesture_dir in category_dir.iterdir():
                if not gesture_dir.is_dir():
                    continue

                gesture_name = gesture_dir.name
                stats["gestures"] += 1
                logger.info(f"  Processing gesture: {gesture_name}")

                keypoints_data = self.extract_from_frames(gesture_dir)

                normalized_data: List[Dict] = []
                for frame_data in keypoints_data:
                    normalized_hands = self.normalize_keypoints(frame_data.get("hands", []))

                    # Count stats
                    stats["total_frames"] += 1
                    if normalized_hands:
                        stats["hands_detected"] += len(normalized_hands)

                    normalized_data.append(
                        {
                            "video_id": frame_data.get("video_id"),
                            "frame_idx": frame_data.get("frame_idx"),
                            "frame_path": frame_data.get("frame_path"),
                            "hands": normalized_hands,
                        }
                    )

                all_data[category_name][gesture_name] = normalized_data

                # Save per-gesture
                gesture_output_dir = output_dir / category_name / gesture_name
                gesture_output_dir.mkdir(parents=True, exist_ok=True)
                with open(gesture_output_dir / "keypoints.json", "w") as f:
                    json.dump(normalized_data, f, indent=2)

        # Save complete dataset
        complete_output = output_dir / "complete_keypoints.json"
        with open(complete_output, "w") as f:
            json.dump(all_data, f, indent=2)

        logger.info(f"Dataset processing complete: {stats}")
        return stats

    def __del__(self):
        if getattr(self, "hands", None) is not None:
            try:
                self.hands.close()
            except Exception:
                pass