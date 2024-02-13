import copy
import gc
import logging
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import tqdm
import wandb
import yaml
from crp.attribution import CondAttribution
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from model_training.correction_methods import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.distance_metrics import cosine_similarities_batch
from utils.latent_features import get_features

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/YOUR_CONFIG.yaml")
    parser.add_argument('--plots', default=False)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file
    config['plots'] = args.plots

    method = config.get("method", "")
    if "clarc" in method.lower():
        measure_quality_cav(config)
    else:
        logger.info(f"Skipping quality-of-CAV metric for method {method}")


def measure_quality_cav(config):
    """ Computes cosine similarity between CAV and actual difference between clean and artifact sample
    Args:
        config (dict): config for model correction run
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name'] + "_attacked" if "attacked" not in config['dataset_name'] else config['dataset_name']
    model_name = config['model_name']

    data_paths = config['data_paths']
    batch_size = config['batch_size']
    artifact_name = config["artifact"]
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    config["device"] = device

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)

    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                              normalize_data=True,
                                              p_artifact=0,
                                              artifact_type=config.get('artifact_type', None),
                                              attacked_classes=[],
                                              **artifact_kwargs, **dataset_specific_kwargs)

    n_classes = len(dataset.class_names)
    ckpt_path = config['ckpt_path']

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    # Construct correction kwargs
    method = config["method"]
    kwargs_correction = {}
   
    if "clarc" in method.lower():
        artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
        mode = "cavs_max"
        kwargs_correction['n_classes'] = len(dataset.class_names)
        kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
        kwargs_correction['sample_ids'] = dataset.idxs_train
        kwargs_correction['mode'] = mode

    correction_method = get_correction_method(method)
    model_corrected = correction_method(copy.deepcopy(model), config, **kwargs_correction)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()
    # mean_length_target = model_corrected.mean_length_targets.item()
    del model_corrected
    torch.cuda.empty_cache();
    gc.collect()
    model.eval()
    attribution = CondAttribution(model)

    results = {}
    for split in [
        # 'train',
        'test',
        'val'
    ]:
        split_set = sets[split]

        dataset_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)

        dataset_artifact_only = dataset_split.get_subset_by_idxs(dataset_split.artifact_ids)
        dataset_artifact_only_clean = dataset_clean_split.get_subset_by_idxs(dataset_split.artifact_ids)

        dl_art = DataLoader(dataset_artifact_only, batch_size=batch_size, shuffle=False)
        dl_clean = DataLoader(dataset_artifact_only_clean, batch_size=batch_size, shuffle=False)

        similarities_all = None
        mean_cav = torch.zeros_like(torch.tensor(cav))
        scores_clean = []
        scores_attacked = []

        diffs = []

        num_samples = 10 if config["plots"] else 0
        high_alignment = []
        high_alignment_imgs = []
        low_alignment = []
        low_alignment_imgs = []

        for (x_att, _), (x_clean, _) in zip(tqdm.tqdm(dl_art), dl_clean):
            # Compute latent representation
            x_latent_att = get_features(x_att.to(device), config, attribution).detach().cpu()
            x_latent_clean = get_features(x_clean.to(device), config, attribution).detach().cpu()

            # Compute similarities between representation difference (attacked-clean) and CAV
            diff_latent = (x_latent_att - x_latent_clean)

            mean_cav += diff_latent.sum(0).reshape(-1) / len(dataset_artifact_only)
            diff_flat = diff_latent.numpy().reshape(diff_latent.shape[0], -1)
            diffs.append(diff_flat)

            similarities = cosine_similarities_batch(diff_flat, cav)
            similarities_all = similarities if similarities_all is None else np.concatenate(
                [similarities_all, similarities])

            def torch2numpy(x):
                std = np.array(dataset.normalize_fn.std)
                mean = np.array(dataset.normalize_fn.mean)
                return x.detach().cpu().permute(0, 2, 3, 1).numpy() * std[None] + mean[None]

            if num_samples:
                similarities_sorted = torch.sort(torch.tensor(similarities), descending=True)
                high_alignment.append(similarities_sorted.values[:num_samples])
                high_alignment_imgs.append(torch2numpy(x_att[similarities_sorted.indices[:num_samples]]))
                similarities_sorted = torch.sort(torch.tensor(similarities).abs(), descending=False)
                low_alignment.append(similarities_sorted.values[:num_samples])
                low_alignment_imgs.append(torch2numpy(x_att[similarities_sorted.indices[:num_samples]]))

            score_attacked = (x_latent_att.flatten(start_dim=1).numpy() * cav[None]).sum(1)
            score_clean = (x_latent_clean.flatten(start_dim=1).numpy() * cav[None]).sum(1)
            scores_clean.append(score_clean)
            scores_attacked.append(score_attacked)

            similarities_all = similarities if similarities_all is None else np.concatenate(
                [similarities_all, similarities])

        ### PLOT high_alignment_imgs and low_alignment_imgs

        if num_samples:

            # concat all images
            high_alignment_imgs = np.concatenate(high_alignment_imgs, axis=0)
            low_alignment_imgs = np.concatenate(low_alignment_imgs, axis=0)

            # concat alignment scores
            high_alignment = np.concatenate(high_alignment, axis=0)
            low_alignment = np.concatenate(low_alignment, axis=0)

            # sort images by alignment
            high_alignment_imgs = high_alignment_imgs[high_alignment.argsort()][::-1][:num_samples]
            low_alignment_imgs = low_alignment_imgs[low_alignment.argsort()][:num_samples]
            high_alignment = high_alignment[high_alignment.argsort()][::-1][:num_samples]
            low_alignment = low_alignment[low_alignment.argsort()][:num_samples]

            fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

            for i in range(num_samples):
                axs[0, i].set_title(f"{high_alignment[i]:.2f}")
                axs[1, i].set_title(f"{low_alignment[i]:.2f}")
                axs[0, i].imshow(high_alignment_imgs[i])
                axs[1, i].imshow(low_alignment_imgs[i])
                axs[0, i].set_xticks([])
                axs[0, i].set_yticks([])
                axs[1, i].set_xticks([])
                axs[1, i].set_yticks([])

            axs[0, 0].set_ylabel("high alignment")
            axs[1, 0].set_ylabel("low alignment")
            plt.tight_layout()
            os.makedirs("results/cav_alignment", exist_ok=True)
            plt.savefig(f"results/cav_alignment/{dataset_name}_{artifact_name}_{split}_alignment.pdf", dpi=300)
            plt.show()

        scores_clean = np.concatenate(scores_clean)
        scores_attacked = np.concatenate(scores_attacked)

        wasserstein_distance = scipy.stats.wasserstein_distance(scores_clean / scores_attacked.mean(),
                                                                scores_attacked / scores_attacked.mean())
        print(wasserstein_distance)

        seperability = (scores_clean.mean() - scores_attacked.mean()) / np.sqrt(
            scores_clean.var() + scores_attacked.var())
        print(seperability)

        results[f"cav_seperability_{artifact_name}_{split}"] = seperability

        seperability = (scores_clean.mean() - scores_attacked.mean()) / np.sqrt(
            scores_clean.std() * scores_attacked.std())

        results[f"cav_seperability2_{artifact_name}_{split}"] = seperability

        thresh = np.linspace(scores_clean.min(), scores_attacked.max(), 1000)
        tpr = []
        fpr = []
        for t in thresh:
            tpr.append((scores_attacked > t).mean())
            fpr.append((scores_clean > t).mean())

        auc = - np.trapz(tpr, fpr)
        print(f"AUC: {auc}")
        results[f"cav_auc_{artifact_name}_{split}"] = auc
        results[f"cav_wassersteindistance_{artifact_name}_{split}"] = wasserstein_distance
        mean_cavs = np.concatenate(diffs, 0) / np.linalg.norm(np.concatenate(diffs, 0).mean(0))
        mean_cav = mean_cavs.mean(0)
        mean_stderr = mean_cavs.std(0) / mean_cavs.shape[0] ** 0.5
        print(split, cosine_similarities_batch(mean_cav[None], cav))
        print(similarities_all[:10])
        mean_mean_cossim = cosine_similarities_batch(mean_cavs, mean_cav[None]).flatten()
        print(f"mean_mean_cossim: {mean_mean_cossim.mean()}")
        mean_cossim = cosine_similarities_batch(mean_cav[None], cav).flatten()
        print(f"mean_cossim: {mean_cossim.mean()}")
        mean_stderr = np.sqrt(np.sum((mean_stderr.flatten() * cav) ** 2)).sum()
        results[f"cav_similarity_{artifact_name}_{split}_mean_cav"] = mean_cossim.mean()
        results[f"cav_similarity_{artifact_name}_{split}_mean_mean_cav"] = mean_mean_cossim.mean()
        results[f"cav_similarity_{artifact_name}_{split}_mean_cav_stderr"] = mean_stderr
        results[f"cav_similarity_{artifact_name}_{split}"] = similarities_all.mean()
        results[f"cav_similarity_{artifact_name}_{split}_stderr"] = similarities_all.std() / similarities_all.shape[
            0] ** 0.5

    if config.get('wandb_api_key', None):
        wandb.log(results)


def get_features(model, batch, config, attribution):
    batch.requires_grad = True
    attr = attribution(batch.to(config["device"]), [], record_layer=[config["layer_name"]])
    features = attr.activations[config["layer_name"]].flatten(start_dim=2).max(2)[0]
    return features


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
