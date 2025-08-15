"""
Minimal timm-backed Vision Transformer wrapper.

This replaces the hand-rolled ViT with a thin wrapper around timm's ViT while
preserving the attributes that SimMIM's code expects (e.g., patch_embed,
cls_token, pos_embed, blocks, norm, head, etc.). It also supports passing
arbitrary model kwargs via a config-provided MODEL_ARGS.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from timm import create_model
from timm.models.layers import trunc_normal_


def _cn_to_dict(maybe_cn: Any) -> Dict[str, Any]:
    try:
        # yacs CfgNode has a .clone()/.to_dict() in some forks; otherwise, recurse
        return {k: _cn_to_dict(v) for k, v in maybe_cn.items()}
    except Exception:
        return maybe_cn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()

        # If an explicit timm model name is given, create via registry; otherwise
        # fall back to constructing a vanilla timm VisionTransformer by kwargs.
        backbone: nn.Module
        resolved_model_args = model_args or {}
        if model_name is not None:
            backbone = create_model(model_name, **resolved_model_args)
            print("Built timm model", model_name, "with args", resolved_model_args)
        else:
            # Conservative subset of kwargs supported by timm VisionTransformer ctor
            allowed_keys = {
                "img_size",
                "patch_size",
                "in_chans",
                "num_classes",
                "embed_dim",
                "depth",
                "num_heads",
                "mlp_ratio",
                "qkv_bias",
                "drop_rate",
                "drop_path_rate",
                "norm_layer",
            }
            ctor_kwargs = {k: v for k, v in legacy_kwargs.items() if k in allowed_keys}
            # Defaults to a common ViT configuration if not provided
            ctor_kwargs.setdefault("img_size", legacy_kwargs.get("img_size", 224))
            ctor_kwargs.setdefault("patch_size", legacy_kwargs.get("patch_size", 16))
            ctor_kwargs.setdefault("in_chans", legacy_kwargs.get("in_chans", 3))
            ctor_kwargs.setdefault("num_classes", legacy_kwargs.get("num_classes", 1000))
            ctor_kwargs.setdefault("embed_dim", legacy_kwargs.get("embed_dim", 768))
            ctor_kwargs.setdefault("depth", legacy_kwargs.get("depth", 12))
            ctor_kwargs.setdefault("num_heads", legacy_kwargs.get("num_heads", 12))
            ctor_kwargs.setdefault("mlp_ratio", legacy_kwargs.get("mlp_ratio", 4.0))
            ctor_kwargs.setdefault("qkv_bias", legacy_kwargs.get("qkv_bias", True))
            ctor_kwargs.setdefault("drop_rate", legacy_kwargs.get("drop_rate", 0.0))
            ctor_kwargs.setdefault("drop_path_rate", legacy_kwargs.get("drop_path_rate", 0.0))

            # Import here to avoid shadowing our wrapper name
            from timm.models.vision_transformer import VisionTransformer as TimmViT

            backbone = TimmViT(**ctor_kwargs)  # type: ignore[arg-type]
            print("Built timm model", model_name, "with args", resolved_model_args)

        # Keep a handle to the backbone and expose expected attributes for SimMIM
        self.backbone = backbone

        # Expose key submodules/parameters to preserve compatibility
        self.patch_embed = getattr(backbone, "patch_embed", None)
        self.cls_token = getattr(backbone, "cls_token", None)
        self.pos_embed = getattr(backbone, "pos_embed", None)
        self.pos_drop = getattr(backbone, "pos_drop", None)
        self.blocks = getattr(backbone, "blocks", None)
        self.norm = getattr(backbone, "norm", None)
        self.head = getattr(backbone, "head", None)

        # Best-effort metadata
        self.embed_dim = getattr(backbone, "embed_dim", None)
        self.num_features = getattr(backbone, "num_features", self.embed_dim)
        # Ensure num_classes attribute exists for SimMIM pretrain assertions
        self.num_classes = getattr(backbone, "num_classes", None)
        if self.num_classes is None:
            # Try to infer from ctor inputs; default to 0 for encoder usage
            inferred = None
            if model_name is not None and "num_classes" in resolved_model_args:
                inferred = resolved_model_args.get("num_classes")
            elif "num_classes" in legacy_kwargs:
                inferred = legacy_kwargs.get("num_classes")
            self.num_classes = 0 if inferred is None else inferred
        self.in_chans = getattr(self.patch_embed, "proj", getattr(self.patch_embed, "proj", None)).in_channels \
            if self.patch_embed is not None and hasattr(self.patch_embed, "proj") else legacy_kwargs.get("in_chans", 3)

        # Patch size as int or tuple
        if self.patch_embed is not None and hasattr(self.patch_embed, "patch_size"):
            ps = getattr(self.patch_embed, "patch_size")
            self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        else:
            self.patch_size = legacy_kwargs.get("patch_size", 16)

        # Relative position bias is not standard in timm ViT; keep a placeholder
        self.rel_pos_bias = getattr(backbone, "rel_pos_bias", None)

    def _trunc_normal_(self, tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
        trunc_normal_(tensor, mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # Optional helpers
    def get_classifier(self):
        if hasattr(self.backbone, "get_classifier"):
            return self.backbone.get_classifier()
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(num_classes, global_pool)
        else:
            if self.embed_dim is None:
                raise RuntimeError("embed_dim unknown; cannot reset classifier")
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, "no_weight_decay"):
            try:
                return set(self.backbone.no_weight_decay())
            except Exception:
                pass
        # Fallback to common tokens if present
        tokens = set()
        if self.pos_embed is not None:
            tokens.add("pos_embed")
        if self.cls_token is not None:
            tokens.add("cls_token")
        return tokens


def build_vit(config) -> nn.Module:
    # Prefer explicit timm model name + args; fallback to legacy fields
    model_name = None
    model_args = {}

    if hasattr(config.MODEL.VIT, "MODEL"):
        model_name = config.MODEL.VIT.MODEL or None
    if hasattr(config.MODEL.VIT, "MODEL_NAME") and not model_name:
        model_name = config.MODEL.VIT.MODEL_NAME or None
    if hasattr(config.MODEL.VIT, "MODEL_ARGS"):
        model_args = _cn_to_dict(config.MODEL.VIT.MODEL_ARGS)

    # Ensure num_classes aligns with top-level setting unless user overrides
    model_args = dict(model_args or {})
    model_args.setdefault("num_classes", config.MODEL.NUM_CLASSES)

    if model_name is None:
        # Legacy path: construct by explicit kwargs subset
        return VisionTransformer(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.VIT.EMBED_DIM,
        depth=config.MODEL.VIT.DEPTH,
        num_heads=config.MODEL.VIT.NUM_HEADS,
        mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        qkv_bias=config.MODEL.VIT.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=nn.LayerNorm,
        )

    return VisionTransformer(model_name=model_name, model_args=model_args)