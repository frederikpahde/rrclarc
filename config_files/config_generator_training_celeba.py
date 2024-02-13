import os
import yaml

config_dir = "training_celeba"
os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 40,
    'device': 'cuda',
    'eval_every_n_epochs': 1,
    'store_every_n_epochs': 5,
    'dataset_name': 'celeba_biased',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'wandb_project_name': 'your_project_name',
    'artifacts_file': 'data/artifacts_bone.json',
    'img_size': 224,
    'milestones': "20,30"
}


def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['celeba_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    config['batch_size'] = 64 if config['model_name'] == 'efficientnet_b4' else 128
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/celeba"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'efficientnet_b0',
    'resnet18',
]:
    base_config['model_name'] = model_name

    lrs = [0.001, 0.005, 0.01] if model_name in ["efficientnet_b0"] else [0.005, 0.001, 0.01]
    for lr in lrs:
        base_config['lr'] = lr
        for pretrained in [
            True,
        ]:

            base_config['pretrained'] = pretrained
            optims = ["adam"] if model_name in ["efficientnet_b0"] else ["sgd"]
            for optim_name in optims:
                base_config['optimizer'] = optim_name
                config_name = f"{model_name}_{optim_name}_lr{lr}_pretrained-{pretrained}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)
