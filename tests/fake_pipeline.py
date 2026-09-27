"""Stand-in for the official video decoder: yields one integer per clip instead of pixels.

Lives in its own module so DataLoader workers (spawn start method) can unpickle it.
"""

import torch
import webdataset as wds


class FakeDecoder(wds.PipelineStage):
    def __init__(self, annotations, skip=frozenset()):
        self.annotations = annotations
        self.skip = skip  # (video_id, start_frame) pairs the decoder drops, like a decode error

    def run(self, src):
        for path in src:
            video_id = path.split("/")[-1].split(".")[0]
            ano = self.annotations[video_id]
            for sf in ano["start_frame"].values:
                if (video_id, int(sf)) in self.skip:
                    continue
                yield dict(video=torch.tensor([int(video_id.split("_")[1]), int(sf)]), verb=0, noun=0, anticipation_time=1.0)
