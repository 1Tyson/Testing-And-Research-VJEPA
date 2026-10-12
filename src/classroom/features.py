"""Frozen V-JEPA 2 encoder -> per-person features for SAV.

The encoder runs on the whole (non-square) frame; RoPE handles the grid. For each person box we keep
BOX_TOKENS tokens of size D, pooled from the token map with RoIAlign:
  - 2x2 RoIAlign cells of the time-averaged map (4 tokens),
  - spatial max inside the box of the time-averaged map (1 token, what mmaction's AVA RoI head uses),
  - box average in each of 4 time segments (4 tokens, keeps the order of motion).
Each keyframe also gets CTX_TOKENS context tokens: mean and max over the whole time-averaged map.
"""

import os

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

BOX_TOKENS = 9
CTX_TOKENS = 2
TIME_SEGMENTS = 4

CHECKPOINTS = {
    "vitl": dict(url="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt", model_name="vit_large"),
    "vith": dict(url="https://dl.fbaipublicfiles.com/vjepa2/vith.pt", model_name="vit_huge"),
    "vitg": dict(url="https://dl.fbaipublicfiles.com/vjepa2/vitg.pt", model_name="vit_giant_xformers"),
    "vitg384": dict(url="https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt", model_name="vit_giant_xformers"),
}


def build_encoder(checkpoint, model_name="vit_large", num_frames=16, resolution=256, checkpoint_key="target_encoder"):
    from vjepa_ek100.model import add_vjepa2_to_path

    add_vjepa2_to_path()
    import src.models.vision_transformer as vit

    model = vit.__dict__[model_name](
        img_size=resolution,
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
        handle_nonsquare_inputs=True,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt[checkpoint_key] if checkpoint_key in ckpt else ckpt
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"encoder weights missing from {os.path.basename(checkpoint)}: {missing[:5]} ...")
    del ckpt, sd
    return model.eval()


@torch.no_grad()
def encode(model, x):
    """x [B, 3, T, H, W] -> token map [B, D, T/2, H/16, W/16] (float32)."""
    B, _, T, H, W = x.shape
    tokens = model(x)
    t, h, w = T // 2, H // 16, W // 16
    return tokens.float().view(B, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()


def _roi(fmap, boxes, batch_idx, out):
    """fmap [N, D, h, w]; boxes normalised [K, 4]; batch_idx [K] -> [K, D, out, out]."""
    h, w = fmap.shape[-2:]
    scale = torch.tensor([w, h, w, h], dtype=fmap.dtype, device=fmap.device)
    rois = torch.cat([batch_idx[:, None].to(fmap.dtype), boxes.to(fmap.dtype) * scale], 1)
    return roi_align(fmap, rois, output_size=out, spatial_scale=1.0, sampling_ratio=2, aligned=True)


@torch.no_grad()
def pool_boxes(fmap, boxes, batch_idx):
    """fmap [B, D, t, h, w]; boxes [K, 4] normalised x1y1x2y2; batch_idx [K]. Returns ([K, 9, D], [B, 2, D])."""
    B, D, t, h, w = fmap.shape
    assert t % TIME_SEGMENTS == 0, f"{t} temporal tokens do not split into {TIME_SEGMENTS} segments"
    mean_map = fmap.mean(2)
    r = _roi(mean_map, boxes, batch_idx, 7)  # [K, D, 7, 7]
    cells = F.adaptive_avg_pool2d(r, 2).flatten(2).transpose(1, 2)  # [K, 4, D]
    peak = r.amax((2, 3))[:, None]  # [K, 1, D]
    seg = fmap.view(B, D, TIME_SEGMENTS, t // TIME_SEGMENTS, h, w).mean(3)  # [B, D, S, h, w]
    seg = seg.permute(0, 2, 1, 3, 4).reshape(B * TIME_SEGMENTS, D, h, w)
    k = len(boxes)
    seg_idx = (batch_idx[:, None] * TIME_SEGMENTS + torch.arange(TIME_SEGMENTS, device=boxes.device)).reshape(-1)
    seg_boxes = boxes[:, None].expand(k, TIME_SEGMENTS, 4).reshape(-1, 4)
    temporal = _roi(seg, seg_boxes, seg_idx, 3).mean((2, 3)).view(k, TIME_SEGMENTS, D)
    box_tok = torch.cat([cells, peak, temporal], 1)
    ctx = torch.stack([mean_map.mean((2, 3)), mean_map.amax((2, 3))], 1)
    return box_tok, ctx
