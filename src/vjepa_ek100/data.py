"""Map-style EK100 val dataset (resumable, shardable) that decodes clips like vjepa2."""

import atexit
import gc

import torch
from torch.utils.data import Dataset, get_worker_info

from .protocol import clip_frame_indices


class EK100ClipDataset(Dataset):
    """Each item is one clip. Consecutive items share a video, so each DataLoader worker
    keeps its last VideoReader open instead of re-indexing an hour-long 1080p file."""

    def __init__(self, clips, transform, frames_per_clip=32, fps=8, anticipation_time=1.0, anchor="official"):
        self.clips = clips.reset_index(drop=True)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.anticipation_time = anticipation_time
        self.anchor = anchor
        self._vr_path = None
        self._vr = None

    def __len__(self):
        return len(self.clips)

    def close(self):
        """Drop the VideoReader while decord's threads are still valid."""
        self._vr = None
        self._vr_path = None
        gc.collect()

    def _reader(self, path):
        if path != self._vr_path:
            from decord import VideoReader, cpu

            self._vr = None
            self._vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
            self._vr_path = path
        return self._vr

    def __getitem__(self, i):
        r = self.clips.iloc[i]
        ok = True
        try:
            vr = self._reader(r.video_path)
            indices = clip_frame_indices(
                r.start_frame,
                r.stop_frame,
                vr.get_avg_fps(),
                frames_per_clip=self.frames_per_clip,
                fps=self.fps,
                anticipation_time=self.anticipation_time,
                anticipation_point=0.0,
                anchor=self.anchor,
            )
            buffer = vr.get_batch(indices).asnumpy()
            video = self.transform(buffer)
        except Exception as e:  # the official loader silently skips such clips; we flag them
            print(f"[warn] clip {r.clip_id} ({r.video_id}) failed: {e!r}")
            ok = False
            self._vr_path = None
            video = torch.zeros(3, self.frames_per_clip, 1, 1)
        return dict(video=video, clip_id=int(r.clip_id), ok=ok)


def worker_init_fn(_):
    """Release decord before interpreter teardown in each DataLoader worker.

    Otherwise a worker can abort on exit with 'pure virtual method called' (seen on Kaggle T4).
    """
    atexit.register(get_worker_info().dataset.close)


def collate(batch):
    ok = [b for b in batch if b["ok"]]
    return dict(
        video=torch.stack([b["video"] for b in ok]) if ok else None,
        clip_id=torch.tensor([b["clip_id"] for b in ok], dtype=torch.long),
        failed=[b["clip_id"] for b in batch if not b["ok"]],
    )
