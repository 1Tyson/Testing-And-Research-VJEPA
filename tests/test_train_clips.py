import numpy as np
import pandas as pd
import pytest

from vjepa_ek100.protocol import build_train_clips, clip_frame_indices, filter_annotations


def fake_train(n_videos=4, per_video=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        vid = f"P{v % 2 + 1:02d}_{101 + v}"
        starts = np.sort(rng.integers(0, 50000, per_video))
        for j, s in enumerate(starts):
            rows.append(dict(narration_id=f"{vid}_{j}", participant_id=vid.split("_")[0], video_id=vid,
                             start_frame=int(s), stop_frame=int(s + rng.integers(30, 400)),
                             verb_class=int(rng.integers(0, 5)), noun_class=int(rng.integers(0, 7))))
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # csv order must not matter


def maps(tdf):
    v, n, a, _ = filter_annotations(tdf, tdf.head(1))
    return v, n, a


def test_ids_do_not_depend_on_available_videos_or_extra_views():
    tdf = fake_train()
    v, n, a = maps(tdf)
    full = build_train_clips(tdf, v, n, a, views=3, seed=1)
    assert len(full) == 3 * len(tdf) and full.clip_id.is_unique
    some = build_train_clips(tdf, v, n, a, {"P01_101": "/x/P01_101.MP4", "P01_103": "/x/P01_103.MP4"}, views=3, seed=1)
    assert set(some.video_id) == {"P01_101", "P01_103"}
    ref = full.set_index("clip_id").loc[some.clip_id]
    for col in ("seg_id", "view", "anticipation_time", "anticipation_point", "action_idx", "start_frame"):
        assert (ref[col].values == some[col].values).all(), col
    fewer = build_train_clips(tdf, v, n, a, views=2, seed=1)
    m = full[full["view"] < 2]
    assert (fewer.anticipation_time.values == m.anticipation_time.values).all()


def test_views_follow_the_official_training_ranges():
    tdf = fake_train(per_video=200)
    v, n, a = maps(tdf)
    c = build_train_clips(tdf, v, n, a, views=4)
    v0 = c[c["view"] == 0]
    assert (v0.anticipation_time == 1.0).all() and (v0.anticipation_point == 0.0).all()
    r = c[c["view"] > 0]
    assert r.anticipation_time.between(0.25, 1.75).all() and r.anticipation_point.between(0.0, 0.25).all()
    assert r.anticipation_time.std() > 0.3
    # label indices follow the class maps
    assert all(a[(x.verb_class, x.noun_class)] == x.action_idx for x in c.itertuples())


def test_view0_matches_the_val_window():
    """View 0 must give exactly the frames of the val setting (1 s, point 0) for both anchors."""
    tdf = fake_train()
    v, n, a = maps(tdf)
    c = build_train_clips(tdf, v, n, a, views=2)
    for r in c[c["view"] == 0].head(20).itertuples():
        for anchor in ("official", "action_start"):
            got = clip_frame_indices(r.start_frame, r.stop_frame, 59.94, anticipation_time=r.anticipation_time,
                                     anticipation_point=r.anticipation_point, anchor=anchor)
            ref = clip_frame_indices(r.start_frame, r.stop_frame, 59.94, anchor=anchor)
            assert (got == ref).all()


def test_dataset_uses_per_clip_anticipation():
    pytest.importorskip("torch")
    from vjepa_ek100.data import EK100ClipDataset

    tdf = fake_train()
    v, n, a = maps(tdf)
    c = build_train_clips(tdf, v, n, a, views=2)
    ds = EK100ClipDataset(c, transform=None)
    r = c[c["view"] == 1].iloc[0]
    assert ds.clip_anticipation(r) == (r.anticipation_time, r.anticipation_point)
    val_like = c.drop(columns=["anticipation_time", "anticipation_point"])
    ds = EK100ClipDataset(val_like, transform=None, anticipation_time=1.0)
    assert ds.clip_anticipation(val_like.iloc[0]) == (1.0, 0.0)
