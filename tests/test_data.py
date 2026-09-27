import numpy as np
import pandas as pd
import pytest
import torch


def test_workers_exit_cleanly_and_bad_clips_are_flagged(vjepa2_root, tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("decord")
    from vjepa_ek100.data import EK100ClipDataset, collate, worker_init_fn
    from vjepa_ek100.model import eval_transform

    path = str(tmp_path / "P01_11.MP4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (320, 240))
    for i in range(600):
        w.write(np.full((240, 320, 3), i % 255, np.uint8))
    w.release()

    # last clip ends past the 600-frame video -> must be reported as failed, not crash
    stops = [300, 350, 400, 450, 500, 900]
    clips = pd.DataFrame(dict(clip_id=range(6), video_id="P01_11", video_path=path, start_frame=[s - 50 for s in stops],
                              stop_frame=stops))
    ds = EK100ClipDataset(clips, eval_transform(64))
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate, worker_init_fn=worker_init_fn)
    ok, failed = [], []
    for b in dl:
        ok += b["clip_id"].tolist()
        failed += b["failed"]
    assert sorted(ok) == [0, 1, 2, 3, 4] and failed == [5]
