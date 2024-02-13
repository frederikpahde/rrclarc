import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import MNIST

from datasets.imagenet import ImageNetDataset, imagenet_mean, imagenet_var, imagenet_augmentation
from utils.artificial_artifact import insert_artifact


def get_imagenet_attacked_dataset(data_paths,
                                  normalize_data=True,
                                  image_size=224,
                                  train=True,
                                  label_map_path=None,
                                  classes=None,
                                  attacked_classes=[],
                                  p_artifact=.5,
                                  artifact_type="ch_text",
                                  **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize(imagenet_mean, imagenet_var))

    transform = T.Compose(fns_transform)


    return ImageNetAttackedDataset(data_paths, train=train, transform=transform, augmentation=imagenet_augmentation,
                           attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                           label_map_path=label_map_path, classes=classes, image_size=image_size, **kwargs)

class ImageNetAttackedDataset(ImageNetDataset):
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
        super().__init__(data_paths, train, transform, augmentation, None, label_map_path, classes,
                         artifact_kwargs.get('subset', None))

        self.attacked_classes = attacked_classes
        self.p_artifact = p_artifact
        self.image_size = image_size
        self.artifact_type = artifact_type
        self.transform_resize = T.Resize((image_size, image_size))
        self.artifact_kwargs = artifact_kwargs

        self.backdoor_prob = artifact_kwargs.get('backdoor_prob', 0)

        np.random.seed(0)
        self.artifact_labels = np.array([wnid in self.attacked_classes and np.random.rand() < self.p_artifact 
                                         for _, wnid in self.samples])
        
        self.artifact_kwargs = artifact_kwargs

        self.artifact_ids = np.where(self.artifact_labels)[0]
        self.sample_ids_by_artifact = {"artificial": self.artifact_ids, artifact_type: self.artifact_ids}
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.artifact_ids))
        if artifact_type == "random_mnist":
            datapath_mnist = self.artifact_kwargs['datapath_mnist']
            data_mnist = MNIST(root=datapath_mnist,
                                    train=True,
                                    download=True,
                                    transform=T.Compose([T.ToTensor(), T.Resize(self.image_size)]))
            self.artifact_kwargs['data_mnist'] = data_mnist



    def add_artifact(self, img, idx):
        random.seed(idx)
        torch.manual_seed(idx)
        np.random.seed(idx)

        return insert_artifact(img, self.artifact_type, **self.artifact_kwargs)
    
    def __getitem__(self, idx):
        path, wnid = self.samples[idx]
        x = Image.open(path).convert('RGB')
        x = self.transform_resize(x)

        insert_backdoor = (np.random.rand() < self.backdoor_prob) and (len(self.attacked_classes) > 0)

        if self.artifact_labels[idx] or insert_backdoor:
            x, _ = self.add_artifact(x, idx)

        x = self.transform(x) if self.transform else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x

        y = self.label_map[wnid]["label"]

        if insert_backdoor:
            y = self.label_map[self.attacked_classes[0]]["label"]

        return x, torch.tensor(y)

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.artifact_labels = self.artifact_labels[np.array(idxs)]

        subset.artifact_ids = np.where(subset.artifact_labels)[0]
        subset.sample_ids_by_artifact = {"artificial": subset.artifact_ids}
        subset.clean_sample_ids = [i for i in range(len(subset)) if i not in subset.artifact_ids]
        return subset

