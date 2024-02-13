import os
import shutil
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
import torch
import yaml
from PIL import Image
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.cav import compute_cav


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--split", default="all")
    parser.add_argument("--layer_name", default="features.28")
    parser.add_argument("--mode", default="cavs_max", choices=['cavs_mean', 'cavs_max', 'crvs'])
    parser.add_argument("--artifact", default="blonde_collar", type=str,
                        choices=["band_aid", "ruler", "skin_marker", "big_l", "small_l"])
    parser.add_argument("--neurons", default=(), type=Tuple[int])
    parser.add_argument("--cav_type", default="signal", type=str)
    parser.add_argument("--save_localization", default=True, type=bool)
    parser.add_argument("--save_examples", default=True, type=bool)
    parser.add_argument('--config_file',
                        default="config_files/YOUR_CONFIG.yaml")
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

    config['batch_size'] = args.batch_size
    config['layer_name'] = args.layer_name
    config['artifact'] = args.artifact

    localize_artifacts(config,
                       split=args.split,
                       mode=args.mode,
                       neurons=args.neurons,
                       save_examples=args.save_examples,
                       save_localization=args.save_localization,
                       cav_type=args.cav_type)


def localize_artifacts(config: dict,
                       split: str,
                       mode: str,
                       neurons: List[int],
                       save_examples: bool,
                       save_localization: bool,
                       cav_type: str):
    """Spatially localize artifacts in input samples.

    Args:
        config (dict): experiment config
        split (str): data split to use
        mode (str): CAV mode
        neurons (List): List of neurons to consider (all if None)
        save_examples (bool): Store example images
        save_localization (bool): Store localization heatmaps
        cav_type (bool): Cav optimizer
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts_file = config.get('artifacts_file', None)

    kwargs_data = {
        "p_artifact": config['p_artifact'],
        "attacked_classes": config['attacked_classes'],
        "artifact_type": config['artifact_type'],
        'label_map_path': config.get('label_map_path', None),
    } if artifacts_file is None else {}
    print(kwargs_data)
    dataset = get_dataset(dataset_name)(data_paths=config['data_paths'],
                                        normalize_data=True,
                                        artifact_ids_file=artifacts_file,
                                        artifact=config['artifact'],
                                        **kwargs_data)

    assert config['artifact'] in dataset.sample_ids_by_artifact.keys(), f"Artifact {config['artifact']} unknown."
    n_classes = len(dataset.class_names)

    artifact_type = config.get('artifact_type', None)
    artifact_extension = f"_{artifact_type}-{config['p_artifact']}" if artifact_type else ""
    artifact_extension += f"-{config['lsb_factor']}" if artifact_type == "lsb" else ""
    path = f"results/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"

    vecs = []

    classes = range(n_classes)
    if "celeba" in dataset_name:
        classes = [0]
    if "isic" in dataset_name:
        classes = [0]
    if "imagenet" in dataset_name:
        classes = [0]
    if "bone" in dataset_name:
        classes = [2]

    sample_ids = []
    for class_id in classes:
        data = torch.load(f"{path}/{config['layer_name']}_class_{class_id}_{split}.pth")
        if data['samples']:
            sample_ids += data['samples']
            vecs.append(torch.stack(data[mode], 0))

    vecs = torch.cat(vecs, 0).to(device)

    # choose only specific neurons
    if neurons:
        vecs = vecs[:, np.array(neurons)]

    sample_ids = np.array(sample_ids)

    all_sample_ids = np.array(dataset.idxs_train)
    filter_sample = np.array([id in all_sample_ids for id in sample_ids])
    vecs = vecs[filter_sample]
    sample_ids = sample_ids[filter_sample]

    artifact_ids = np.array(
        [id_ for id_ in dataset.sample_ids_by_artifact[config['artifact']] if np.argwhere(sample_ids == id_)])
    target_ids = np.array(
        [np.argwhere(sample_ids == id_)[0][0] for id_ in artifact_ids])
    print(f"Chose {len(target_ids)} target samples.")

    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.class_names),
                                                       ckpt_path=config['ckpt_path'])
    model = model.to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    attribution = CondAttribution(model)

    img_to_plt = lambda x: dataset.reverse_normalization(x.detach().cpu()).permute((1, 2, 0)).int().numpy()
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    targets = np.array([1 * (j in target_ids) for j, x in enumerate(sample_ids)])

    X = vecs.detach().cpu().clamp(min=0).numpy()
    print("Fitting linear model..")

    w = compute_cav(X, targets, cav_type=cav_type)[..., None, None].to(device)

    samples = [dataset[sample_ids[i]] for i in target_ids]
    data_sample = torch.stack([s[0] for s in samples]).to(device).requires_grad_()
    target = [s[1] for s in samples]

    conditions = [{"y": t.item()} for t in target]

    batch_size = 32
    num_batches = int(np.ceil(len(data_sample) / batch_size))

    heatmaps = []
    inp_imgs = []

    layer_name = config['layer_name']

    for b in tqdm(range(num_batches)):
        data = data_sample[batch_size * b: batch_size * (b + 1)]
        attr = attribution(data,
                           conditions[batch_size * b: batch_size * (b + 1)],
                           composite, record_layer=[layer_name])
        act = attr.activations[layer_name]

        inp_imgs.extend([img_to_plt(s.detach().cpu()) for s in data])

        attr = attribution(data, [{}], composite, start_layer=layer_name, init_rel=act.clamp(min=0) * w)
        heatmaps.extend([hm_to_plt(h.detach().cpu().clamp(min=0)) for h in attr.heatmap])

    if save_examples:
        num_imgs = min(len(inp_imgs), 72) * 2
        grid = int(np.ceil(np.sqrt(num_imgs) / 2) * 2)

        fig, axs_ = plt.subplots(grid, grid, dpi=150, figsize=(grid * 1.2, grid * 1.2))

        for j, axs in enumerate(axs_):
            ind = int(j * grid / 2)
            for i, ax in enumerate(axs[::2]):
                if len(inp_imgs) > ind + i:
                    ax.imshow(inp_imgs[ind + i])
                    ax.set_xlabel(f"sample {int(artifact_ids[ind + i])}", labelpad=1)
                ax.set_xticks([])
                ax.set_yticks([])

            for i, ax in enumerate(axs[1::2]):
                if len(inp_imgs) > ind + i:
                    max = np.abs(heatmaps[ind + i]).max()
                    ax.imshow(heatmaps[ind + i], cmap="bwr", vmin=-max, vmax=max)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"artifact", labelpad=1)

        plt.tight_layout(h_pad=0.1, w_pad=0.0)

        os.makedirs(f"results/localization/{dataset_name}/{model_name}", exist_ok=True)
        plt.savefig(
            f"results/localization/{dataset_name}/{model_name}/{config['artifact']}_{layer_name}_{mode}_{cav_type}.pdf")
        plt.show()

    if save_localization:
        path = f"data/localized_artifacts/{dataset_name}/{config['artifact']}"
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        for i in range(len(heatmaps)):
            sample_id = int(artifact_ids[i])
            heatmap = heatmaps[i]
            heatmap[heatmap < 0] = 0
            heatmap = heatmap / heatmap.max() * 255
            im = Image.fromarray(heatmap).convert("L")
            im.save(f"{path}/{sample_id}.png")


if __name__ == "__main__":
    main()
