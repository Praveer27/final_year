import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _first_hand_vec(frame: dict) -> np.ndarray:
    """
    Convert one frame's hands data to a (63,) vector (21 landmarks * 3 coords).
    If no hand detected, returns zeros.
    """
    hands = frame.get("hands") or []
    if not hands:
        return np.zeros((63,), dtype=np.float32)

    kps = (hands[0] or {}).get("keypoints") or []
    if len(kps) != 21:
        return np.zeros((63,), dtype=np.float32)

    arr = np.array([[kp["x"], kp["y"], kp["z"]] for kp in kps], dtype=np.float32)  # (21,3)
    return arr.reshape(-1)  # (63,)


class GestureSequenceDataset(Dataset):
    """
    Builds samples as (sequence, label) where:
      - sequence shape: (seq_len, 63)
      - label: int class id
    Samples are grouped by (category, gesture, video_id).
    """

    def __init__(self, json_path: str, seq_len: int = 45):
        self.seq_len = seq_len
        with open(json_path, "r") as f:
            data = json.load(f)

        # gesture -> label_id
        self.label_map = {}
        self.samples = []

        label_id = 0
        for category, gestures in data.items():
            for gesture, frames in gestures.items():
                if gesture not in self.label_map:
                    self.label_map[gesture] = label_id
                    label_id += 1

                # group frames by video_id
                by_video = {}
                for fr in frames:
                    vid = fr.get("video_id", "unknown")
                    by_video.setdefault(vid, []).append(fr)

                # build one sequence per video_id
                for vid, vframes in by_video.items():
                    vframes.sort(key=lambda x: x.get("frame_idx", -1))

                    seq = [_first_hand_vec(fr) for fr in vframes]

                    # pad / truncate
                    if len(seq) >= self.seq_len:
                        seq = seq[: self.seq_len]
                    else:
                        pad = [np.zeros((63,), dtype=np.float32) for _ in range(self.seq_len - len(seq))]
                        seq = seq + pad

                    x = np.stack(seq, axis=0)  # (T,63)
                    y = self.label_map[gesture]
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)