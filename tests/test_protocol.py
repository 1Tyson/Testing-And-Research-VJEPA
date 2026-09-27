import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vjepa_ek100.protocol import (
    build_val_clips,
    clip_frame_indices,
    filter_annotations,
    index_videos,
    official_eval_multiplicity,
)


def _synthetic_ek100(tmp_path, n_train=400, n_val_videos=11, seed=0):
    rng = np.random.default_rng(seed)
    tdf = pd.DataFrame(
        dict(
            video_id=[f"P01_{i % 5:03d}" for i in range(n_train)],
            verb_class=rng.integers(0, 20, n_train),
            noun_class=rng.integers(0, 40, n_train),
            start_frame=rng.integers(0, 10000, n_train),
        )
    )
    rows = []
    for v in range(n_val_videos):
        n = int(rng.integers(1, 30))  # very uneven clips per video, like EK100
        starts = rng.choice(100000, n, replace=False)
        for s in starts:
            rows.append(dict(video_id=f"P02_{100 + v}", verb_class=rng.integers(0, 22), noun_class=rng.integers(0, 42),
                             start_frame=int(s), stop_frame=int(s) + 100))
    vdf = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # csv not sorted by start_frame
    tdf["stop_frame"] = tdf["start_frame"] + 100
    for vid in set(vdf.video_id) - {"P02_101"}:  # one val video missing on disk
        d = tmp_path / vid.split("_")[0] / "videos"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{vid}.MP4").touch()
    tdf.to_csv(tmp_path / "train.csv", index=False)
    vdf.to_csv(tmp_path / "val.csv", index=False)
    return tdf, vdf


def test_frame_indices():
    idx = clip_frame_indices(600, 1200, 60.0)
    assert len(idx) == 32 and idx[0] == 1140 - 224 and idx[-1] == 1140 - 7 and np.all(np.diff(idx) == 7)
    idx = clip_frame_indices(600, 1200, 60.0, anchor="action_start")
    assert idx[-1] == 540 - 7
    assert clip_frame_indices(10, 50, 50.0)[0] == 0  # padded with first frame


def test_filter_annotations_matches_official(vjepa2_root, tmp_path):
    from evals.action_anticipation_frozen.epickitchens import filter_annotations as official

    tdf, vdf = _synthetic_ek100(tmp_path)
    ref = official(str(tmp_path), str(tmp_path / "train.csv"), str(tmp_path / "val.csv"), file_format=0)
    verbs, nouns, actions, vdf_f = filter_annotations(pd.read_csv(tmp_path / "train.csv"), pd.read_csv(tmp_path / "val.csv"))
    assert verbs == ref["verbs"] and nouns == ref["nouns"] and actions == ref["actions"]

    clips = build_val_clips(vdf_f, verbs, nouns, actions, index_videos(str(tmp_path)))
    paths, annos = ref["val"]
    ref_order = [(os.path.basename(p)[:-4], sf) for p in paths for sf in annos[os.path.basename(p)[:-4]].start_frame]
    assert list(zip(clips.video_id, clips.start_frame)) == ref_order
    assert "P02_101" not in set(clips.video_id)


@pytest.mark.parametrize("world_size,batch_size,num_workers", [(3, 2, 2), (4, 3, 2), (2, 2, 0)])
def test_multiplicity_matches_official_loop(vjepa2_root, tmp_path, world_size, batch_size, num_workers):
    """Run the real webdataset + DataLoader pipeline rank by rank with the official
    `ipe`-bounded, restart-on-exhaustion loop and compare clip counts."""
    from evals.action_anticipation_frozen.epickitchens import filter_annotations as official
    from evals.action_anticipation_frozen.epickitchens import get_video_wds_dataset
    from fake_pipeline import FakeDecoder

    _synthetic_ek100(tmp_path, seed=world_size)
    ref = official(str(tmp_path), str(tmp_path / "train.csv"), str(tmp_path / "val.csv"), file_format=0)
    paths, annos = ref["val"]
    num_clips = sum(len(a) for a in annos.values())
    ipe = num_clips // (world_size * batch_size)

    seen = Counter()
    for rank in range(world_size):
        _, info = get_video_wds_dataset(
            batch_size=batch_size, input_shards=paths, video_decoder=FakeDecoder(annos), training=False,
            world_size=world_size, rank=rank, num_workers=num_workers, persistent_workers=False, pin_memory=False,
        )
        it = iter(info.dataloader)
        for _ in range(ipe):
            try:
                b = next(it)
            except Exception:
                it = iter(info.dataloader)
                try:
                    b = next(it)
                except Exception:
                    break  # empty rank
            seen.update((f"P02_{int(v)}", int(s)) for v, s in b[0].tolist())

    verbs, nouns, actions, vdf_f = filter_annotations(pd.read_csv(tmp_path / "train.csv"), pd.read_csv(tmp_path / "val.csv"))
    clips = build_val_clips(vdf_f, verbs, nouns, actions, index_videos(str(tmp_path)))
    key = dict(zip(clips.clip_id, zip(clips.video_id, clips.start_frame)))
    ours = Counter({key[c]: n for c, n in official_eval_multiplicity(clips, world_size, batch_size, num_workers).items()})
    assert ours == seen
