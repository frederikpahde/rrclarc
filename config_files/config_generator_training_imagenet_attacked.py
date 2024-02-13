import json
import os
import numpy as np
import yaml

config_dir = "training_imagenet"

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("../data/label-map-imagenet.json", "r") as f:
    label_map = json.load(f)

wnids = np.array(list(label_map.keys()))
rng = np.random.default_rng(42)
wnids_subset = [f"{s}" for s in sorted(list(rng.choice(wnids, 50, replace=False)))]

attacked_classes = [f"{rng.choice(wnids_subset)}"]

print(wnids_subset)
print(attacked_classes)

base_config = {
    'num_epochs': 20,
    'device': 'cuda',
    'eval_every_n_epochs': 1,
    'store_every_n_epochs': 2,
    'dataset_name': 'imagenet_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'label_map_path': 'data/label-map-imagenet.json',
    'img_size': 224,
    'classes': None,
    'attacked_classes': ["n01440764"],
    'p_artifact': 0.5,
    'percentage_batches': .2
}

print(base_config)


def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['imagenet_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    config['batch_size'] = 64 if config['model_name'] == 'efficientnet_b4' else 128
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/imagenet"]

    if config['artifact_type'] == "random_mnist":
        config['datapath_mnist'] = "/mnt/mnist"

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'vgg16',
    'efficientnet_b0',
    'resnet18',
]:
    for artifact_type in [
        "ch_time",
    ]:
        for backdoor_prob in [0.001]:
            base_config['backdoor_prob'] = backdoor_prob
        
            base_config['model_name'] = model_name
            base_config['artifact_type'] = artifact_type

            for time_format in [
                "datetime"
                ]:
                base_config['time_format'] = time_format

                lrs = [0.0001] if model_name in ["efficientnet_b0"] else [0.0005]
                for lr in lrs:
                    base_config['lr'] = lr
                    for pretrained in [
                        True,
                    ]:
                        for start_epoch in [0,
                                            ]:

                            base_config[
                                "ckpt_path"] = f"/mnt/models/checkpoint_{model_name}_last.pth" if start_epoch > 0 else None
                            base_config['pretrained'] = pretrained
                            base_config['start_epoch'] = start_epoch
                            optims = ["adam"] if model_name in ["efficientnet_b0"] else ["sgd"]
                            for optim_name in optims:
                                base_config['optimizer'] = optim_name

                                config_name = f"backdoor{backdoor_prob}-{artifact_type}-{time_format}_{base_config['p_artifact']}_{model_name}_{optim_name}_lr{lr}_pretrained-{pretrained}_epoch{start_epoch}"

                                store_local(base_config, config_name)
                                store_cluster(base_config, config_name)
