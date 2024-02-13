import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.celeba import CelebADataset

celeba_augmentation = T.Compose([
    # T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.5),
    # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25),
    T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(224)], p=.25)
])


def get_celeba_hm_dataset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, artifact=None,
                          **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebAHmDataset(data_paths, train=True, transform=transform, augmentation=celeba_augmentation,
                           artifact_ids_file=artifact_ids_file, artifact=artifact)


class CelebAHmDataset(CelebADataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, artifact=None,
                 artifact_ids_file=None):
        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)
        assert len(data_paths) == 1, "Only 1 path accepted for Bone Dataset"

        self.hm_path = f"data/localized_artifacts/celeba"
        artifact_paths = glob.glob(f"{self.hm_path}/{artifact}/*")
        artifact_sample_ids = np.array([int(x.split("/")[-1].split(".")[0]) for x in artifact_paths])
        self.artifact_ids = artifact_sample_ids
        self.hms = ["" for _ in range(len(self.metadata))]
        for i, j in enumerate(artifact_sample_ids):
            path = artifact_paths[i]
            if self.hms[j]:
                self.hms[j] += f",{path}"
            else:
                self.hms[j] += f"{path}"

        self.metadata["hms"] = self.hms

    def __getitem__(self, i):
        image, bone_age = super().__getitem__(i)

        if self.metadata["hms"].loc[i]:
            # print(self.hms[item].split(","))
            # TODO: LOOKS GOOD FOR MULTIPLE ARTIFACTS OR SHOULD NORMALIZE EACH?
            heatmaps = torch.stack(
                [torch.tensor(np.asarray(Image.open(hm))) for hm in self.metadata["hms"].loc[i].split(",")]).clamp(
                min=0).sum(0).float()
        else:
            heatmaps = torch.zeros_like(image[0])

        return image, bone_age, heatmaps
