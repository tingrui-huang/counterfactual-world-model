"""Decision ① — frame preprocessing: grayscale, crop to the play area, resize.

The exact crop/scale lives in config.FrameSpec so every px <-> token conversion
downstream is exact. The resize is itself a coarsening (84/160 = 0.525: the paddle's
4.59 px/step becomes ~2.41 resized px) — the first suspect if Stage 2 fails.

The HARD invariant (skill): this preprocessing is applied to MODEL-facing frames AND
to the frames we encode as eval targets (oracle CF frames pass through the SAME eye,
which needs the same input format) — but the oracle itself replays at full pixel
fidelity and its 4-number ground truth is never coarsened.
"""
import numpy as np
from PIL import Image

from pong_counterfactual.cjepa_rung3.config import FRAME, FrameSpec

# ITU-R 601 luma — matches ALE's own getScreenGrayscale convention
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def preprocess(rgb: np.ndarray, spec: FrameSpec = FRAME) -> np.ndarray:
    """(210,160,3) uint8 RGB -> (res,res) uint8 grayscale, cropped + resized."""
    gray = (rgb.astype(np.float32) @ _LUMA)                  # (210,160)
    gray = gray[spec.crop_top:spec.crop_bottom, :]           # (160,160) play area
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    img = img.resize((spec.res, spec.res), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def spec_dict(spec: FrameSpec = FRAME) -> dict:
    """The exact preprocessing parameters, stored alongside every dataset."""
    return {"crop_top": spec.crop_top, "crop_bottom": spec.crop_bottom,
            "res": spec.res, "scale": spec.scale,
            "gray": "ITU-R601", "resample": "bilinear"}
