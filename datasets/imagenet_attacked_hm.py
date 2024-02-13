import random
import numpy as np
import torch
from datasets.imagenet import imagenet_mean, imagenet_var, imagenet_augmentation
import torchvision.transforms as T
from datasets.imagenet_attacked import ImageNetAttackedDataset
from utils.artificial_artifact import insert_artifact
from PIL import Image


def get_imagenet_attacked_hm_dataset(data_paths, 
                                  normalize_data=True,
                                  image_size=224,
                                  train=True,
                                  label_map_path=None,
                                  classes=None,
                                  attacked_classes=[],
                                  p_artifact=.5,
                                  artifact_type="ch_text",
                                  **kwargs):
    print("in get dataset_hm", label_map_path, kwargs)
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize(imagenet_mean, imagenet_var))

    transform = T.Compose(fns_transform)


    return ImageNetAttackedHmDataset(data_paths, train=train, transform=transform, augmentation=imagenet_augmentation,
                           attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                           label_map_path=label_map_path, classes=classes, image_size=image_size, **kwargs)

class ImageNetAttackedHmDataset(ImageNetAttackedDataset):
    def __init__(self,
                 data_paths,
                 train,
                 transform,
                 augmentation,
                 attacked_classes=[],
                 p_artifact=.5,
                 artifact_type="ch_text",
                 label_map_path=None,
                 classes=None,
                 image_size=224,
                 **artifact_kwargs):
        super().__init__(data_paths, train, transform, augmentation, attacked_classes, p_artifact,
                         artifact_type, label_map_path, classes, image_size, **artifact_kwargs)

    
    def __getitem__(self, idx):
        path, wnid = self.samples[idx]
        x = Image.open(path).convert('RGB')
        x = self.transform_resize(x)

        if self.artifact_labels[idx]:
            x, mask = self.add_artifact(x, idx)
        else:
            mask = torch.zeros((self.image_size, self.image_size))

        x = self.transform(x) if self.transform else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x
        y = self.label_map[wnid]["label"]

        return x, y, mask.type(torch.uint8)
