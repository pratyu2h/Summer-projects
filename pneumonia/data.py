""" Dataset loader for NIH ChestX-ray14. """
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from model import NIH_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NIHChestXrayDataset(Dataset):
    def __init__(self, data_root: Path | str, split_file: str, transform=None,
                 csv_name: str = "Data_Entry_2017.csv", image_subdir: str = "images"):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / image_subdir
        self.transform = transform

        meta = pd.read_csv(self.data_root / csv_name)
        meta = meta.set_index("Image Index")

        with open(self.data_root / split_file) as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        self.filenames = [fn for fn in self.filenames if fn in meta.index]

        self.labels = torch.zeros(len(self.filenames), len(NIH_CLASSES), dtype=torch.float32)
        for i, fn in enumerate(self.filenames):
            findings = str(meta.loc[fn, "Finding Labels"])
            if findings == "No Finding":
                continue
            for label in findings.split("|"):
                label = label.strip()
                if label in NIH_CLASSES:
                    self.labels[i, NIH_CLASSES.index(label)] = 1.0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_dir / self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        elif not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        return image, self.labels[idx]

    def class_positive_counts(self) -> torch.Tensor:
        """Positive count per class -- useful for pos_weight / focal alpha."""
        return self.labels.sum(dim=0)


def get_transforms(train: bool = True, image_size: int = 224):
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    """Smoke testing"""
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "images").mkdir()
        rows = []
        for i in range(6):
            fn = f"{i:08d}_000.png"
            Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(tmp / "images" / fn)
            findings = "No Finding" if i % 3 == 0 else "Cardiomegaly|Effusion" if i % 3 == 1 else "Hernia"
            rows.append({"Image Index": fn, "Finding Labels": findings})
        pd.DataFrame(rows).to_csv(tmp / "Data_Entry_2017.csv", index=False)
        with open(tmp / "train_val_list.txt", "w") as f:
            f.write("\n".join(r["Image Index"] for r in rows))

        ds = NIHChestXrayDataset(tmp, "train_val_list.txt", transform=get_transforms(train=False, image_size=64))
        assert len(ds) == 6
        img, label = ds[1]
        assert img.shape == (3, 64, 64)
        assert label[NIH_CLASSES.index("Cardiomegaly")] == 1.0
        assert label[NIH_CLASSES.index("Effusion")] == 1.0
        assert label.sum() == 2.0
        no_finding_label = ds[0][1]
        assert no_finding_label.sum() == 0.0
        counts = ds.class_positive_counts()
        assert counts.shape == (len(NIH_CLASSES),)
        print(f"[ok] dataset len={len(ds)}, sample label sum={label.sum().item()}, class_positive_counts sum={counts.sum().item()}")
