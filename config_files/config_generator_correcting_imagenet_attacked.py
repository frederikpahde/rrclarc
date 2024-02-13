import copy
import json
import os

import numpy as np
import yaml

config_dir = "correcting_imagenet"
os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("../data/label-map-imagenet.json", "r") as f:
    label_map = json.load(f)

wnids = np.array(list(label_map.keys()))
rng = np.random.default_rng(42)
wnids_subset = [f"{s}" for s in sorted(list(rng.choice(wnids, 50, replace=False)))]
wnids = [f"{s}" for s in wnids]
attacked_classes = [f"{rng.choice(wnids_subset)}"]

_base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'imagenet_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'img_size': 224,
    'wandb_project_name': 'your_project_name',
    'label_map_path': 'data/label-map-imagenet.json',
    'classes': wnids,
    'subset_correction': 100,
    'attacked_classes': ["n01440764"],
    'p_artifact': 0.5,
    "limit_train_batches": 1000,
    "artifact": "artificial",
    'eval_acc_every_epoch': False,
    'unique_wandb_ids': True,
}

models_ckpts = {
    "vgg16": "checkpoints/checkpoint_vgg16_imagenet_attacked_ch_time.pth",
    "resnet18": "checkpoints/checkpoint_resnet18_imagenet_attacked_ch_time.pth",
    "efficientnet_b0": "checkpoints/checkpoint_efficientnet_b0_imagenet_attacked_ch_time.pth",
}
def store_local(config, config_name):
    config['ckpt_path'] = models_ckpts[model_name]
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_paths'] = [local_config['imagenet_dir']]
    config['batch_size'] = local_config['local_batch_size']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    config['ckpt_path'] = models_ckpts[model_name]
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_paths'] = ["/mnt/imagenet"]
    config['batch_size'] = 64

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


layer_names_by_model = {
}

for model_name, layer_name in [
    ('vgg16', 'features.28'),
    ('resnet18', 'last_conv'),
    ('efficientnet_b0', 'last_conv'),
]:
    _base_config['model_name'] = model_name
    _base_config['layer_name'] = layer_name
    for artifact_type in [
        "ch_time"
    ]:

        _base_config["artifact_type"] = artifact_type
        base_config = copy.deepcopy(_base_config)

        base_config['time_format'] = "datetime"

        for artifact in [
            "artificial"
        ]:
            base_config['artifact'] = artifact
            for lr in [
                0.0001,

            ]:
                base_config['lr'] = lr

                optim_name = "adam" if model_name == "efficientnet_b0" else "sgd"

                base_config['optimizer'] = optim_name

                ### VANILLA
                config_vanilla = copy.deepcopy(base_config)
                method = 'Vanilla'
                config_vanilla['method'] = method
                config_vanilla['lamb'] = 0.0
                config_name = f"{model_name}_{method}_{layer_name}"
                store_local(config_vanilla, config_name)
                store_cluster(config_vanilla, config_name)

                config_vanilla = copy.deepcopy(base_config)
                method = 'Vanilla'
                config_name = f"{model_name}_{method}"
                config_vanilla['num_epochs'] = 0
                config_vanilla['method'] = method
                store_cluster(config_vanilla, config_name)
                store_local(config_vanilla, config_name)

                #
                method = 'RRR_ExpLogSum'
                for lamb in [
                    0.05, 0.1,
                    0.5, 1.0,
                    5, 10,
                    50, 100,
                    500, 1000,
                    5000, 10000,
                    50000, 100000,
                    500000, 1000000,
                    5000000, 10000000,
                ]:
                    base_config['method'] = method
                    base_config['lamb'] = lamb

                    config_name = f"{artifact_type}-{base_config['p_artifact']}_{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                    store_local(base_config, config_name)
                    store_cluster(base_config, config_name)

                for direction_mode in [
                    "svm",
                    "signal",
                    "ridge",
                    "logistic",
                    "lasso"
                ]:
                    base_config['direction_mode'] = direction_mode

                    for method in [
                        'AClarc',
                        'PClarc'
                    ]:
                        base_config['method'] = method
                        lamb = 1.0
                        base_config['lamb'] = lamb
                        config_name = f"{model_name}_{method}_{direction_mode}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                        store_local(base_config, config_name)
                        store_cluster(base_config, config_name)

                for direction_mode in [
                    'signal',
                    # "svm",
                    # "lasso",
                    # "ridge",
                    # "logistic",
                ]:
                    base_config['direction_mode'] = direction_mode
                    for lamb2 in [
                        "cosine_mean",
                        "l2_mean",
                        "l1_mean",
                    ]:

                        base_config['compute'] = lamb2

                        for lamb in [
                            0.05, 0.1,
                            0.5, 1.0,
                            5, 10,
                            50, 100,
                            500, 1000,
                            5000, 10000,
                            50000, 100000,
                            500000, 1000000,
                            5000000, 10000000,
                            5000000, 100000000,
                        ]:
                            base_config['lamb'] = lamb
                            for method in [
                                'RRClarc'
                            ]:
                                base_config['method'] = method
                                for criterion in [
                                    # 'all',
                                    'allrand',
                                    # 'logprobs',
                                ]:
                                    base_config['target_class'] = 0
                                    base_config['criterion'] = criterion
                                    base_config['cav_type'] = "cavs_max"
                                    config_name = f"{model_name}_{method}_{direction_mode}_{optim_name}_lr{lr}_lamb{lamb}_comp{lamb2}_crit{criterion}_{artifact}_{layer_name}"
                                    store_local(base_config, config_name)
                                    store_cluster(base_config, config_name)
