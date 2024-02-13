import os
import yaml

config_dir = "training_bone"

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 100,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 20,
    'dataset_name': 'bone_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'img_size': 224,
    'attacked_classes': [2],
    'p_artifact': 0.2
}


def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['bone_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    config['batch_size'] = 64 if config['model_name'] == 'efficientnet_b4' else 128
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/bone"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'efficientnet_b0',
    'resnet18',
]:
    base_config['model_name'] = model_name

    lrs = [0.001, 0.005, 0.0005] if model_name in ["efficientnet_b0"] else [0.005, 0.001]
    for lr in lrs:
        base_config['lr'] = lr
        for pretrained in [
            True,
        ]:
            base_config['pretrained'] = pretrained

            optims = ["adam"] if model_name in ["efficientnet_b0"] else ["sgd"]
            for optim_name in optims:
                for artifact_type in ['white_color']:
                    base_config['artifact_type'] = artifact_type

                    base_config['optimizer'] = optim_name
                    config_name = f"attacked_{artifact_type}_{model_name}_{optim_name}_lr{lr}_pretrained-{pretrained}"
                    store_local(base_config, config_name)
                    store_cluster(base_config, config_name)
