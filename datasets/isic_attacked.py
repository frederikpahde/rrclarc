import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import MNIST

from datasets.isic import ISICDataset, isic_augmentation
from utils.artificial_artifact import insert_artifact


def get_isic_attacked_dataset(data_paths, normalize_data=True, image_size=224,
                              attacked_classes=[], p_artifact=.5, artifact_type='ch_text', **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return ISICAttackedDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                               attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                               image_size=image_size, **kwargs)


class ISICAttackedDataset(ISICDataset):
    def __init__(self,
                 data_paths,
                 train=False,
                 transform=None,
                 augmentation=None,
                 binary_target=False,
                 attacked_classes=[],
                 p_artifact=.5,
                 artifact_type="ch_text",
                 image_size=224,
                 **artifact_kwargs):

        super().__init__(data_paths, train, transform, augmentation, binary_target, None)

        self.attacked_classes = attacked_classes
        self.p_artifact = p_artifact
        self.image_size = image_size
        self.artifact_type = artifact_type
        self.transform_resize = T.Resize((image_size, image_size))
        self.artifact_kwargs = artifact_kwargs

        if not self.artifact_kwargs:
            self.artifact_kwargs = {
                'text': "Clever Hans",
                'fill': (0, 0, 0),
                'img_size': self.image_size
            } if self.artifact_type == "ch_text" else {
                'value': 100,
                'channel': 0,
                'op_type': 'add'
            }
        
        if artifact_type == "random_mnist":
            datapath_mnist = self.artifact_kwargs['datapath_mnist']
            data_mnist = MNIST(root=datapath_mnist,
                                    train=True,
                                    download=True,
                                    transform=T.Compose([T.ToTensor(), T.Resize(self.image_size)]))
            self.artifact_kwargs['data_mnist'] = data_mnist


        np.random.seed(0)
        self.artifact_labels = np.array(
            [(np.array([self.metadata.loc[idx][cl] for cl in self.attacked_classes]) == 1.0).any() and
             np.random.rand() < self.p_artifact
             for idx in range(len(self))])
        self.artifact_ids = np.where(self.artifact_labels)[0]
        self.sample_ids_by_artifact = {"artificial": self.artifact_ids, artifact_type: self.artifact_ids}
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids]

    def add_artifact(self, img, idx):
        random.seed(idx)
        torch.manual_seed(idx)
        np.random.seed(idx)

        return insert_artifact(img, self.artifact_type, **self.artifact_kwargs)

    def __getitem__(self, i):
        row = self.metadata.loc[i]

        path = self.train_dirs_by_version[row.version] if self.train else self.test_dirs_by_version[row.version]
        img = Image.open(path / Path(row['image'] + '.jpg'))
        img = self.transform_resize(img)
        if self.artifact_labels[i]:
            img, _ = self.add_artifact(img, i)

        img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)
        columns = self.metadata.columns.to_list()
        target = torch.Tensor([columns.index(row[row == 1.0].index[0]) - 1 if self.train else 0]).long()[0]
        return img, target

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.artifact_labels = self.artifact_labels[np.array(idxs)]

        subset.artifact_ids = np.where(subset.artifact_labels)[0]
        subset.sample_ids_by_artifact = {"artificial": subset.artifact_ids}
        subset.clean_sample_ids = [i for i in range(len(subset)) if i not in subset.artifact_ids]
        return subset
