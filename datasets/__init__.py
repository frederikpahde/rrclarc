import logging
from typing import Callable

from datasets.bone import get_bone_dataset
from datasets.bone_attacked import get_bone_attacked_dataset
from datasets.bone_attacked_hm import get_bone_attacked_hm_dataset
from datasets.bone_hm import get_bone_hm_dataset
from datasets.celeba import get_celeba_dataset
from datasets.celeba_biased import get_celeba_biased_dataset
from datasets.celeba_biased_hm import get_celeba_biased_hm_dataset
from datasets.celeba_hm import get_celeba_hm_dataset
from datasets.imagenet import get_imagenet_dataset
from datasets.imagenet_attacked import get_imagenet_attacked_dataset
from datasets.imagenet_attacked_hm import get_imagenet_attacked_hm_dataset
from datasets.isic import get_isic_dataset
from datasets.isic_attacked import get_isic_attacked_dataset
from datasets.isic_attacked_hm import get_isic_attacked_hm_dataset
from datasets.isic_hm import get_isic_hm_dataset

logger = logging.getLogger(__name__)

DATASETS = {
    "imagenet": get_imagenet_dataset,
    "imagenet_attacked": get_imagenet_attacked_dataset,
    "imagenet_attacked_hm": get_imagenet_attacked_hm_dataset,
    "isic": get_isic_dataset,
    "isic_hm": get_isic_hm_dataset,
    "isic_attacked": get_isic_attacked_dataset,
    "isic_attacked_hm": get_isic_attacked_hm_dataset,
    "bone": get_bone_dataset,
    "bone_hm": get_bone_hm_dataset,
    "bone_attacked": get_bone_attacked_dataset,
    "bone_attacked_hm": get_bone_attacked_hm_dataset,
    "celeba": get_celeba_dataset,
    "celeba_biased": get_celeba_biased_dataset,
    "celeba_hm": get_celeba_hm_dataset,
    "celeba_biased_hm": get_celeba_biased_hm_dataset,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
    
def get_dataset_kwargs(config):
    dataset_specific_kwargs = {
        "label_map_path": config["label_map_path"],
        "classes": config["classes"],
        "backdoor_prob": config.get("backdoor_prob", 0),
        "train": True
    } if "imagenet" in config['dataset_name'] else {}

    return dataset_specific_kwargs
