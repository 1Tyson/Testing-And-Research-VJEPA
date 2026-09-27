"""Build the frozen V-JEPA 2 encoder+predictor and the released EK100 attentive probe.

Requires the official repo on sys.path (see `add_vjepa2_to_path`).
"""

import os
import sys

import torch


def add_vjepa2_to_path(root=None):
    root = root or os.environ.get("VJEPA2_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "evals")):
        raise RuntimeError("Set VJEPA2_ROOT (or --vjepa2_root) to a clone of facebookresearch/vjepa2")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def eval_transform(resolution):
    from evals.action_anticipation_frozen.dataloader import make_transforms

    return make_transforms(training=False, crop_size=resolution)


def build_backbone(cfg, checkpoint, device):
    from evals.action_anticipation_frozen.models import init_module

    data = cfg["experiment"]["data"]
    mk = cfg["model_kwargs"]
    return init_module(
        module_name=mk["module_name"],
        device=device,
        frames_per_clip=data["frames_per_clip"],
        frames_per_second=data["frames_per_second"],
        resolution=data["resolution"],
        checkpoint=checkpoint,
        model_kwargs=mk["pretrain_kwargs"],
        wrapper_kwargs=mk["wrapper_kwargs"],
    )


def build_probe(cfg, num_verbs, num_nouns, num_actions, embed_dim, device, probe_checkpoint=None):
    from evals.action_anticipation_frozen.models import AttentiveClassifier

    c = cfg["experiment"]["classifier"]
    probe = AttentiveClassifier(
        verb_classes=range(num_verbs),
        noun_classes=range(num_nouns),
        action_classes=range(num_actions),
        embed_dim=embed_dim,
        num_heads=c["num_heads"],
        depth=c["num_probe_blocks"],
        use_activation_checkpointing=False,
    )
    if probe_checkpoint is not None:
        ckpt = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt["classifiers"][0] if "classifiers" in ckpt else ckpt
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        # Shape mismatch here almost always means a different train csv -> different classes.
        for k, v in probe.state_dict().items():
            if k in sd and sd[k].shape != v.shape:
                raise RuntimeError(
                    f"probe key {k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(v.shape)}. "
                    "Use the full official EPIC_100_train.csv so class counts match."
                )
        probe.load_state_dict(sd, strict=True)
    return probe.to(device).eval()
