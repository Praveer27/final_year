import torch
import numpy as np


def load_checkpoint(ckpt_path: str, create_model_fn, model_kwargs: dict, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    label_map = ckpt["label_map"]              # gesture -> id
    id_to_label = {v: k for k, v in label_map.items()}

    model = create_model_fn(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len = ckpt.get("seq_len", 45)
    input_size = ckpt.get("input_size", 63)

    return model, id_to_label, seq_len, input_size


@torch.no_grad()
def predict_sequence(model, seq: np.ndarray, device="cpu"):
    """
    seq: (T, 63) float32
    returns: probs (num_classes,)
    """
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,63)
    logits = model(x)  # (1,C)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs