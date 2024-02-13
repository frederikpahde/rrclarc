import copy

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import CelebA

from datasets.base_dataset import BaseDataset

celeba_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.5),
    # T.RandomVerticalFlip(p=.5),
    # T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25),
    T.RandomApply(transforms=[T.Pad(10, fill=-(46.9 / 255.) / (22.6 / 255.)), T.Resize(224)], p=.25)
])


def get_celeba_dataset(data_paths, normalize_data=True, image_size=224, artifact_ids_file=None, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    return CelebADataset(data_paths, train=True, transform=transform, augmentation=celeba_augmentation,
                         artifact_ids_file=artifact_ids_file)


class CelebADataset(BaseDataset):
    def __init__(self, data_paths, train=True, transform=None, augmentation=None, artifact_ids_file=None):
        super().__init__(data_paths, train, transform, augmentation, artifact_ids_file)
        assert len(data_paths) == 1, "Only 1 path accepted for Bone Dataset"

        ds = CelebA(root=data_paths[0], split='all', download=False, transform=transform)
        self.path = f"{data_paths[0]}/{ds.base_folder}"
        ATTR = 'Blond_Hair'
        ATTR2 = 'Wearing_Necktie'
        attr_id = np.where(np.array(ds.attr_names) == ATTR)[0][0]
        attr_id2 = np.where(np.array(ds.attr_names) == ATTR2)[0][0]
        USE_SUBSET = True
        if USE_SUBSET:
            print("Using subset")
            NTH = 10
        else:
            NTH = 1
        filter_indices = np.zeros(len(ds.attr))
        filter_indices[::NTH] = 1
        print(f"Chosen attribute {ATTR} with id {attr_id}.")
        labels = ds.attr[:, attr_id]
        labels2 = ds.attr[:, attr_id2]
        both = np.where(np.logical_and(labels, labels2) == 1)[0]
        # print(both)
        filter_indices[both] = 1

        both_names = np.array(ds.filename)[both]
        # print("', '".join(both_names) + "\n")

        labels = ds.attr[:, attr_id][filter_indices == 1]
        labels2 = ds.attr[:, attr_id2][filter_indices == 1]
        both = np.where(np.logical_and(labels, labels2) == 1)[0]
        # print(both)

        labels_not_blonde = 1 * (ds.attr[:, attr_id][filter_indices == 1] == 0)
        labels_collar = ds.attr[:, attr_id2][filter_indices == 1]
        not_blonde_collar = np.where(np.logical_and(labels_not_blonde, labels_collar) == 1)[0]

        print(f"Chosen attribute {ATTR} with id {attr_id}.")
        self.metadata = pd.DataFrame(
            {'image_id': np.array(ds.filename)[filter_indices == 1], 'targets': labels})

        self.normalize_fn = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.classes = [f'Non-{ATTR}', ATTR]
        self.class_names = self.classes

        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.var = torch.Tensor([0.5, 0.5, 0.5])

        self.weights = self.compute_weights(np.array([len(labels) - labels.sum(), labels.sum()]))

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, .1)

        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()
        if USE_SUBSET:
            print("Using subset")
            # transfer all artifacts to test set
            artifacts_in_train = [x for x in both if x in self.idxs_train]
            artifacts_to_keep = artifacts_in_train[::NTH]
            artifacts_to_remove = [x for x in artifacts_in_train if x not in artifacts_to_keep]
            self.idxs_train = np.array([x for x in self.idxs_train if x not in artifacts_to_remove])
            # select half of the artifacts to be in val and half in test
            artifacts_to_remove = np.array(artifacts_to_remove)
            np.random.default_rng(42).shuffle(artifacts_to_remove)
            self.idxs_val = np.concatenate([self.idxs_val, artifacts_to_remove[::2]])
            self.idxs_test = np.concatenate([self.idxs_test, artifacts_to_remove[1::2]])
            self.idxs_test.sort()
            self.idxs_val.sort()

        print(f"Artifacts in train: {len([x for x in both if x in self.idxs_train])}")
        print(f"Artifacts in test: {len([x for x in both if x in self.idxs_test])}")
        print(f"Artifacts in val: {len([x for x in both if x in self.idxs_val])}")
        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

    def get_all_ids(self):
        return list(self.metadata['image_id'].values)

    #
    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        img_name = f"{self.path}/img_align_celeba/{self.metadata.iloc[idx]['image_id']}"
        image = Image.open(img_name).convert("RGB")

        target = torch.tensor(self.metadata.iloc[idx]["targets"])

        if self.transform:
            image = self.transform(image)

        if self.do_augmentation:
            image = self.augmentation(image)

        return image.float(), target

    def get_sample_name(self, i):
        return self.metadata.iloc[i]['image_id']

    def get_target(self, i):
        target = torch.tensor(self.metadata.iloc[i]["targets"])
        return target

    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset


if __name__ == "__main__":
    ds = CelebADataset(["datasets"], train=True, transform=None, augmentation=None)
