"""Decode SAV clips and sample the frames around a keyframe, like mmaction's SampleAVAFrames (test mode).

The SAV authors extract frames with `ffmpeg -r 30`, so frame i (0-based) is the frame shown at time i / 30 s.
A keyframe at timestamp t uses center frame fps * (t - timestamp_start) and `num_frames` frames spaced by `stride`,
clipped to the clip ([0, fps * (timestamp_end - timestamp_start) - 1]).
"""

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def keyframe_indices(ts, num_frames=16, stride=4, fps=30, timestamp_start=0, timestamp_end=3):
    center = int(round(fps * (ts - timestamp_start)))
    idx = center - (num_frames // 2) * stride + stride * np.arange(num_frames)
    return np.clip(idx, 0, fps * (timestamp_end - timestamp_start) - 1)


def decode_at_fps(path, fps=30):
    """All frames of a clip resampled to `fps` (nearest earlier frame, as `ffmpeg -r` does). uint8 [N, H, W, 3]."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames, times = [], []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.pts * stream.time_base))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    times = np.asarray(times) - times[0]
    order = np.argsort(times, kind="stable")
    times = times[order]
    n_out = max(1, int(np.floor(times[-1] * fps + 1e-6)) + 1)
    grid = np.arange(n_out) / fps
    src = np.clip(np.searchsorted(times, grid + 1e-6, side="right") - 1, 0, len(times) - 1)
    return np.stack([frames[order[i]] for i in src])


def target_size(h, w, short_side=256, multiple=16):
    """Keep the aspect ratio, short side = `short_side`, both sides multiples of the patch size."""
    if h <= w:
        nh, nw = short_side, w * short_side / h
    else:
        nh, nw = h * short_side / w, short_side
    return int(round(nh / multiple)) * multiple, int(round(nw / multiple)) * multiple


def to_model_input(frames, size):
    """uint8 [T, H, W, 3] -> normalised float [3, T, h, w]."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).permute(1, 0, 2, 3).contiguous()
