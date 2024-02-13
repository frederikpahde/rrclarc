import copy
import os
import yaml

config_dir = "correcting_bone_attacked"

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'bone_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'your_api_key',
    'img_size': 224,
    'wandb_project_name': 'your_project_name',
    'attacked_classes': [2],
    'p_artifact': .2,
    'artifact_type': 'white_color',
    'eval_acc_every_epoch': False,
    'unique_wandb_ids': True,
}


def store_local(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"{local_config['checkpoint_dir']}/checkpoint_{model_name}_attacked_bone_last.pth"
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_paths'] = [local_config['bone_dir']]
    config['batch_size'] = local_config['local_batch_size']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"checkpoints/checkpoint_{model_name}_attacked_bone_last.pth"
    config['ckpt_path_corrected'] = f"checkpoints/{config_name}/last.ckpt"
    if "Vanilla" in config_name and config['num_epochs'] == 0:
        config['ckpt_path_corrected'] = config['ckpt_path']
    config['data_paths'] = ["/mnt/bone"]
    config['batch_size'] = 64

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name, layer_name in [
    ('vgg16', 'features.28'),
    ('resnet18', 'last_conv'),
    ('efficientnet_b0', 'last_conv'),
]:

    base_config['model_name'] = model_name
    base_config['layer_name'] = layer_name

    for artifact in [
        "artificial",
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
            config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_{artifact}"
            store_cluster(config_vanilla, config_name)
            store_local(config_vanilla, config_name)

            config_vanilla = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_vanilla['method'] = method
            config_vanilla['lamb'] = 0.0
            config_vanilla['num_epochs'] = 0
            config_name = f"{model_name}_{method}"
            store_cluster(config_vanilla, config_name)
            store_local(config_vanilla, config_name)
            #
            method = 'RRR_ExpLogSum'
            for lamb in [
                0.0005, 0.001,
                0.005, 0.01,
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
                # base_config['num_epochs'] = 5
                config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_lamb{lamb}_{artifact}_{layer_name}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)

            for direction_mode in [
                *[
                    "signal",
                    "svm",
                    "lasso",
                    "ridge",
                    "logistic"
                ]
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
                # 'svm',
                # 'lasso',
                # 'ridge',
                # 'logistic',
                'signal'
            ]:
                base_config['direction_mode'] = direction_mode

                for lamb2 in [
                    # "l1_mean",
                    "l2_mean",
                    # "cosine_mean"
                ]:

                    base_config['compute'] = lamb2

                    for lamb in [
                        0.0005, 0.001,
                        0.005, 0.01,
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
                                base_config['criterion'] = criterion
                                for cav_type in [
                                    "cavs_max",
                                ]:
                                    base_config['cav_type'] = cav_type
                                    config_name = f"{model_name}_{method}_{cav_type}_{direction_mode}_{optim_name}_lr{lr}_lamb{lamb}_comp{lamb2}_crit{criterion}_{artifact}_{layer_name}"
                                    store_local(base_config, config_name)
                                    store_cluster(base_config, config_name)
