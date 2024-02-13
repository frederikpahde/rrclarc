import copy
import glob
import json
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset

imagenet_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])

imagenet_mean = torch.Tensor((0.485, 0.456, 0.406))
imagenet_var = torch.Tensor((0.229, 0.224, 0.225))


def get_imagenet_dataset(data_paths,
                         normalize_data=True,
                         image_size=224,
                         artifact_ids_file=None,
                         train=True,
                         label_map_path=None,
                         classes=None,
                         subset=None,
                         **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize(imagenet_mean, imagenet_var))

    transform = T.Compose(fns_transform)

    return ImageNetDataset(data_paths, train=train, transform=transform, augmentation=imagenet_augmentation,
                           artifact_ids_file=artifact_ids_file, label_map_path=label_map_path, classes=classes,
                           subset=subset)

def extract_wnid(path):
    return path.split("/")[-1]

class ImageNetDataset(BaseDataset):
    def __init__(self,
                 data_paths,
                 train=False,
                 transform=None,
                 augmentation=None,
                 artifact_ids_file=None,
                 label_map_path=None,
                 classes=None,
                 subset=None):
        #print("classes", classes)
        #print(data_paths)
        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)
        split = "train" if train else "val"
        path = f"{data_paths[0]}/{split}"
        #print(path)

        if subset is not None:
            self.samples = self.read_samples(path, classes[0:1])
            self.samples += self.read_samples(path, classes[1:])[::subset]
        else:
            self.samples = self.read_samples(path, classes)
        self.normalize_fn = T.Normalize(imagenet_mean, imagenet_var)

        assert label_map_path is not None, "label_map_path required for ImageNet"
        with open(label_map_path) as file:
            self.label_map = json.load(file)

        self.classes = [class_details["name"] for wnid, class_details in self.label_map.items() if
                        classes is None or wnid in classes]
        self.class_names = self.classes

        if classes:
            self.label_map = {k: v for k, v in self.label_map.items() if k in classes}
            for idx, wnid in enumerate(classes):
                self.label_map[wnid]['label'] = torch.tensor(idx).long()
            # print(f"Updated label map to {self.label_map}")

        counts = Counter([self.label_map[wnid]["name"] for _, wnid in self.samples])
        dist = torch.Tensor([counts[wnid] for wnid in self.classes])

        self.mean = imagenet_mean
        self.var = imagenet_var

        self.weights = self.compute_weights(dist)

        # We split the training set into 90/10 train/val splits and use the official val set as test set
        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, 0)

        # Add test samples
        num_samples_before = len(self.samples)
        if subset is not None:
            self.samples += self.read_samples(f"{data_paths[0]}/val", classes[0:1])
            self.samples += self.read_samples(f"{data_paths[0]}/val", classes[1:])[::subset]
        else:
            self.samples += self.read_samples(f"{data_paths[0]}/val", classes)
        self.idxs_test = np.arange(num_samples_before, len(self.samples))
        #print(f"Train: {len(self.idxs_train)}, Val: {len(self.idxs_val)}, Test: {len(self.idxs_test)}")
        #print(f"Val idxs: {self.idxs_val}")
    
    def read_samples(self, path, classes):
        samples = []
        for subdir in sorted(glob.glob(f"{path}/*")):
            wnid = extract_wnid(subdir)
            if classes is None or wnid in classes:
                for path in sorted(glob.glob(f"{subdir}/*.JPEG")):
                    samples.append((path, wnid))
        return samples
            


    def __len__(self):
        return len(self.samples)
    
    def get_target(self, idx):
        _, wnid = self.samples[idx]
        return self.label_map[wnid]["label"]
    
    def __getitem__(self, idx):
        path, wnid = self.samples[idx]
        x = Image.open(path).convert('RGB')

        x = self.transform(x) if self.transform else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x
        y = self.label_map[wnid]["label"]

        return x, y
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.samples = [subset.samples[i] for i in idxs]
        return subset