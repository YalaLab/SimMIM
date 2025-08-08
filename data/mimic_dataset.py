import os
from typing import List, Optional, Tuple, Any

import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


def get_mimic_abs_path(img_meta: dict, root: str) -> str:
    subject_id = str(img_meta["subject_id"])  # e.g., "11"
    study_id = img_meta["study_id"]
    dicom_id = img_meta["dicom_id"]
    return os.path.join(
        root,
        f"p{subject_id[:2]}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg",
    )


class MimicCXRImagesSimple(VisionDataset):
    """
    Minimal MIMIC-CXR dataset for SimMIM pre-training.

    - Reads a JSON file (JSON or JSONL) describing samples, with each record containing
      a key (default: 'images') that holds a list whose first element is a dict with
      'subject_id', 'study_id', 'dicom_id'.
    - Returns the transformed image and a dummy target (0), so downstream code can
      ignore labels while retaining the expected structure.
    """

    def __init__(
        self,
        json_path: str,
        root: str,
        img_paths_key: str = "images",
        transform=None,
    ) -> None:
        super().__init__(root=root, transforms=None, transform=transform, target_transform=None)
        self.root_dir = root

        # Load the split definition
        # Supports JSON lines and regular JSON arrays
        with open(json_path, "r") as f:
            try:
                # pandas can read JSON arrays or dicts. If it's JSONL, lines=True works.
                df = pd.read_json(f, lines=True)
            except ValueError:
                f.seek(0)
                df = pd.read_json(f)

        self._paths: List[str] = [
            get_mimic_abs_path(item[0], root) for item in df[img_paths_key]
        ]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path = self._paths[index]
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError as e:
            raise RuntimeError(f"Image not found: {path}") from e

        if self.transform is not None:
            img = self.transform(img)

        # Return a dummy target to preserve ((img, mask), target) structure downstream
        return img, 0

