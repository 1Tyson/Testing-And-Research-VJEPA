import pickle

import numpy as np
import pytest

from classroom.ava_metric import average_precision, frame_map, iou_matrix
from classroom.sav import build_box_table, multi_hot, read_annotations, read_labelmap, read_proposals


def _voc_ap_reference(scores, tp, num_gt):
    """Plain re-statement of object_detection.utils.metrics (compute_precision_recall + compute_average_precision)."""
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    precision = ctp / (ctp + cfp)
    recall = ctp / num_gt
    r = np.concatenate([[0], recall, [1]])
    p = np.concatenate([[0], precision, [0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    i = np.where(r[1:] != r[:-1])[0] + 1
    return np.sum((r[i] - r[i - 1]) * p[i])


def test_average_precision_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.random(50)
        tp = rng.random(50) < 0.4
        n = tp.sum() + 3
        assert average_precision(s, tp, n) == pytest.approx(_voc_ap_reference(s, tp, n))


def test_iou():
    a = np.array([[0, 0, 0.5, 0.5]])
    b = np.array([[0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0.5], [0.6, 0.6, 1, 1]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_frame_map_perfect_and_swapped():
    gt = {
        "v,0001": [(np.array([0, 0, 0.5, 1.0]), [1]), (np.array([0.5, 0, 1.0, 1.0]), [2])],
        "v,0002": [(np.array([0, 0, 0.5, 1.0]), [1, 2])],
    }
    boxes1 = np.array([[0, 0, 0.5, 1.0], [0.5, 0, 1.0, 1.0]])
    det = {"v,0001": (boxes1, np.array([[0.9, 0.1], [0.1, 0.9]])), "v,0002": (boxes1[:1], np.array([[0.8, 0.8]]))}
    m, ap = frame_map(gt, det, [1, 2])
    assert m == pytest.approx(1.0) and ap == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}

    # Swap the scores of keyframe 1: each class ranks a false positive first.
    det["v,0001"] = (boxes1, np.array([[0.1, 0.9], [0.9, 0.1]]))
    m, _ = frame_map(gt, det, [1, 2])
    assert m < 1.0


def test_frame_map_duplicate_detection_is_false_positive_and_excluded_keys_still_count():
    gt = {"v,0001": [(np.array([0, 0, 1, 1]), [1])], "v,0002": [(np.array([0, 0, 1, 1]), [1])]}
    det = {"v,0001": (np.array([[0, 0, 1, 1], [0, 0, 1, 1]]), np.array([[0.9], [0.8]]))}
    # One TP then one duplicate FP; the excluded keyframe's box still counts in the GT total -> recall 0.5.
    _, ap = frame_map(gt, det, [1], excluded={"v,0002"})
    assert ap[1] == pytest.approx(0.5)


def _write_sav(tmp_path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    (ann / "train.csv").write_text(
        "clipA_1_3,1,0.1,0.1,0.3,0.5,2,0\n"
        "clipA_1_3,1,0.1,0.1,0.3,0.5,5,0\n"
        "clipA_1_3,1,0.5,0.1,0.7,0.5,1,1\n"
        "clipA_1_3,2,0.1,0.1,0.3,0.5,2,0\n"
    )
    (ann / "train_excluded_timestamps.csv").write_text("clipA_1_3,2\n")
    (ann / "education_first_label.pbtxt").write_text(
        'label {\n  name: "hand-raising"\n  label_id: 1\n  label_type: PERSON_MOVEMENT\n}\n'
        'label {\n  name: "reading"\n  label_id: 2\n}\nitem {\n  name: "writing"\n  id: 5\n}\n'
    )
    with open(ann / "train_person_det.pkl", "wb") as f:
        pickle.dump({"clipA_1_3,0001": np.array([[0.1, 0.1, 0.3, 0.5, 0.95], [0.4, 0.1, 0.6, 0.5, 0.5]]),
                     "clipA_1_3,0002": np.array([[0.1, 0.1, 0.3, 0.5, 0.3]])}, f)
    return ann


def test_annotations_and_box_table(tmp_path):
    ann = _write_sav(tmp_path)
    assert read_labelmap(ann / "education_first_label.pbtxt") == {1: "hand-raising", 2: "reading", 5: "writing"}
    gt = read_annotations(ann / "train.csv")
    assert sorted(gt) == ["clipA_1_3,0001", "clipA_1_3,0002"]
    assert sorted(lab for _, lab in gt["clipA_1_3,0001"]) == [[1], [2, 5]]

    props = read_proposals(ann / "train_person_det.pkl", score_thr=0.8)
    assert len(props["clipA_1_3,0001"][0]) == 1  # 0.5 < 0.8 dropped
    assert len(props["clipA_1_3,0002"][0]) == 1  # best box kept when all are under the threshold

    df = build_box_table(str(ann), "train", with_proposals=True)
    assert set(df.key) == {"clipA_1_3,0001"}  # excluded keyframe dropped
    assert (df.source == "gt").sum() == 2 and (df.source == "prop").sum() == 1
    y = multi_hot(df.labels.tolist(), [1, 2, 5])
    assert y[df.source.eq("gt").to_numpy()].sum() == 3


def test_keyframe_indices_and_size():
    pytest.importorskip("torch")
    from classroom.video import keyframe_indices, target_size

    idx = keyframe_indices(1, num_frames=16, stride=4)
    assert idx[0] == 0 and idx[8] == 30 and idx[-1] == 58
    assert keyframe_indices(3).max() == 89
    assert target_size(720, 1280) == (256, 448)
    assert target_size(1080, 1920, short_side=384) == (384, 688)


def test_pool_boxes_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from classroom.features import BOX_TOKENS, CTX_TOKENS, pool_boxes

    fmap = torch.randn(2, 8, 8, 16, 28)
    boxes = torch.tensor([[0.1, 0.1, 0.3, 0.6], [0.5, 0.2, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0]])
    box_tok, ctx = pool_boxes(fmap, boxes, torch.tensor([0, 0, 1]))
    assert box_tok.shape == (3, BOX_TOKENS, 8) and ctx.shape == (2, CTX_TOKENS, 8)
    # A map that only changes over time: every box token of a time segment equals that segment's mean.
    t_only = torch.randn(1, 8, 8, 1, 1).expand(1, 8, 8, 16, 28).contiguous()
    box_tok, _ = pool_boxes(t_only, boxes[1:2], torch.tensor([0]))
    seg_mean = t_only[0, :, :, 0, 0].view(8, 4, 2).mean(2).T
    torch.testing.assert_close(box_tok[0, 5:], seg_mean)
    torch.testing.assert_close(box_tok[0, 4], t_only[0, :, :, 0, 0].mean(1))
